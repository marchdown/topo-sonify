use crate::audio::oscillator::{Oscillator, Waveform};

/// MorseSynth: a synthesizer driven by Morse theory on the complex.
///
/// ## Mathematical Foundation
///
/// Morse theory studies how the topology of a space changes as we
/// sweep a "height function" f: M → ℝ across a manifold M.
///
/// At each critical point p where ∇f(p) = 0, the topology changes:
///   - Index 0 (minimum): a new component is born → β₀ increases
///   - Index 1 (saddle):  a loop closes or a component merges
///   - Index 2 (maximum): a void closes → β₂ decreases
///
/// The **Morse inequalities** relate the number of critical points
/// cₖ of index k to the Betti numbers:
///   cₖ ≥ βₖ  for all k
///   Σ (-1)ᵏ cₖ = χ(M) = Σ (-1)ᵏ βₖ
///
/// ## Sonification Strategy
///
/// We define a "height function" on a simplicial complex via the
/// filtration parameter. As we sweep through the filtration:
///
/// - **Births (index 0)**: trigger a note-on event, frequency
///   determined by the filtration value
/// - **Saddles (index 1)**: modulate existing voices, bending
///   frequencies toward each other (components merging)
/// - **Deaths (index 2+)**: trigger note-off with decay time
///   proportional to the feature's persistence
///
/// The result: the filtration literally plays the topology as a
/// temporal musical score.
pub struct MorseSynth {
    voices: Vec<MorseVoice>,
    sample_rate: f64,
    time: f64,
    events: Vec<MorseEvent>,
    event_index: usize,
    playback_speed: f64,
}

#[derive(Debug, Clone)]
struct MorseVoice {
    osc: Oscillator,
    envelope: ADSREnvelope,
    active: bool,
    dimension: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MorseEvent {
    pub time: f64, // when to trigger (seconds)
    pub event_type: MorseEventType,
    pub frequency: f64,
    pub dimension: usize,
    pub persistence: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum MorseEventType {
    Birth,
    Death,
}

/// Simple ADSR envelope.
#[derive(Debug, Clone)]
struct ADSREnvelope {
    attack: f64,
    decay: f64,
    sustain: f64,
    release: f64,
    state: EnvelopeState,
    level: f64,
    time_in_state: f64,
    sample_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EnvelopeState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl ADSREnvelope {
    fn new(attack: f64, decay: f64, sustain: f64, release: f64, sample_rate: f64) -> Self {
        Self {
            attack,
            decay,
            sustain,
            release,
            state: EnvelopeState::Idle,
            level: 0.0,
            time_in_state: 0.0,
            sample_rate,
        }
    }

    fn trigger(&mut self) {
        self.state = EnvelopeState::Attack;
        self.time_in_state = 0.0;
    }

    fn release(&mut self) {
        self.state = EnvelopeState::Release;
        self.time_in_state = 0.0;
    }

    fn next_sample(&mut self) -> f64 {
        let dt = 1.0 / self.sample_rate;
        self.time_in_state += dt;

        match self.state {
            EnvelopeState::Idle => {
                self.level = 0.0;
            }
            EnvelopeState::Attack => {
                self.level = (self.time_in_state / self.attack).min(1.0);
                if self.time_in_state >= self.attack {
                    self.state = EnvelopeState::Decay;
                    self.time_in_state = 0.0;
                }
            }
            EnvelopeState::Decay => {
                let t = (self.time_in_state / self.decay).min(1.0);
                self.level = 1.0 + (self.sustain - 1.0) * t;
                if self.time_in_state >= self.decay {
                    self.state = EnvelopeState::Sustain;
                    self.time_in_state = 0.0;
                }
            }
            EnvelopeState::Sustain => {
                self.level = self.sustain;
            }
            EnvelopeState::Release => {
                let t = (self.time_in_state / self.release).min(1.0);
                self.level = self.sustain * (1.0 - t);
                if self.time_in_state >= self.release {
                    self.state = EnvelopeState::Idle;
                    self.level = 0.0;
                }
            }
        }

        self.level
    }

    fn is_active(&self) -> bool {
        self.state != EnvelopeState::Idle
    }
}

impl MorseSynth {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            voices: Vec::new(),
            sample_rate,
            time: 0.0,
            events: Vec::new(),
            event_index: 0,
            playback_speed: 1.0,
        }
    }

    /// Build events from persistence pairs.
    /// Each pair (birth, death, dim) becomes a birth event and a death event.
    /// Frequency mapping: filtration_value → frequency via logarithmic mapping.
    pub fn from_persistence_pairs(
        pairs: &[(f64, f64, usize)],
        freq_range: (f64, f64),
        time_scale: f64,
        sample_rate: f64,
    ) -> Self {
        let mut events = Vec::new();

        // Find filtration range
        let max_filt = pairs
            .iter()
            .map(|&(_, d, _)| if d.is_infinite() { 0.0 } else { d })
            .fold(0.0_f64, f64::max);
        let max_filt = if max_filt == 0.0 { 1.0 } else { max_filt };

        for &(birth, death, dim) in pairs {
            // Map filtration value to frequency (log scale)
            let t = birth / max_filt;
            let freq = freq_range.0 * (freq_range.1 / freq_range.0).powf(t);

            // Map to time
            let birth_time = birth * time_scale;
            let persistence = if death.is_infinite() {
                max_filt
            } else {
                death - birth
            };

            events.push(MorseEvent {
                time: birth_time,
                event_type: MorseEventType::Birth,
                frequency: freq / (1.0 + dim as f64), // lower octaves for higher dim
                dimension: dim,
                persistence,
            });

            if !death.is_infinite() {
                events.push(MorseEvent {
                    time: death * time_scale,
                    event_type: MorseEventType::Death,
                    frequency: freq / (1.0 + dim as f64),
                    dimension: dim,
                    persistence,
                });
            }
        }

        // Sort events by time
        events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

        Self {
            voices: Vec::new(),
            sample_rate,
            time: 0.0,
            events,
            event_index: 0,
            playback_speed: 1.0,
        }
    }

    pub fn set_playback_speed(&mut self, speed: f64) {
        self.playback_speed = speed.max(0.01);
    }

    pub fn next_sample(&mut self) -> f64 {
        let dt = 1.0 / self.sample_rate * self.playback_speed;

        // Process events at current time
        while self.event_index < self.events.len()
            && self.events[self.event_index].time <= self.time
        {
            let event = self.events[self.event_index];
            match event.event_type {
                MorseEventType::Birth => {
                    let waveform = match event.dimension {
                        0 => Waveform::Sine,
                        1 => Waveform::Triangle,
                        2 => Waveform::Sawtooth,
                        _ => Waveform::Square,
                    };
                    let osc = Oscillator::new(waveform, event.frequency, self.sample_rate);
                    let release = (event.persistence * 2.0).clamp(0.1, 5.0);
                    let mut env = ADSREnvelope::new(0.01, 0.1, 0.7, release, self.sample_rate);
                    env.trigger();
                    self.voices.push(MorseVoice {
                        osc,
                        envelope: env,
                        active: true,
                        dimension: event.dimension,
                    });
                }
                MorseEventType::Death => {
                    // Release the oldest voice at this frequency/dimension
                    for voice in &mut self.voices {
                        if voice.active
                            && voice.dimension == event.dimension
                            && (voice.osc.frequency - event.frequency).abs() < 1.0
                        {
                            voice.envelope.release();
                            voice.active = false;
                            break;
                        }
                    }
                }
            }
            self.event_index += 1;
        }

        // Generate audio from all voices
        let mut output = 0.0;
        for voice in &mut self.voices {
            let env = voice.envelope.next_sample();
            if env > 0.0 {
                output += voice.osc.next_sample() * env * 0.2;
            }
        }

        // Clean up dead voices
        self.voices.retain(|v| v.envelope.is_active());

        self.time += dt;
        output.clamp(-1.0, 1.0)
    }

    pub fn fill_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.next_sample();
        }
    }

    /// Reset playback to the beginning.
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.event_index = 0;
        self.voices.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morse_synth_from_pairs() {
        let pairs = vec![(0.0, 0.5, 0), (0.1, 0.8, 0), (0.3, 0.6, 1)];
        let mut synth = MorseSynth::from_persistence_pairs(&pairs, (110.0, 880.0), 1.0, 44100.0);
        let mut buffer = vec![0.0; 44100]; // 1 second
        synth.fill_buffer(&mut buffer);
        let energy: f64 = buffer.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }
}
