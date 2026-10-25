//! Drum Pads' samples: `mix.pad_load` makes an audio file (decoded into a session source at the
//! session rate, saved with the session like any other audio) or an existing session source the
//! sample a pad plays; `mix.pad_clear` empties a pad. The track's instrument becomes Drum Pads
//! when it is something else. Pad settings (tune, decay, low-pass, level, pan, reverse, start,
//! end, choke group) are `mix.instrument_param` values `pad<N>_<setting>`.

use super::*;
use crate::cmd;
use serde_json::json;
use soundcraft_model::{Insert, SourceId, TrackKind};
use std::path::Path;

const PADS: &str = "drum_pads";
/// Pads Drum Pads plays (notes 36 up).
const PAD_COUNT: u64 = 16;
/// Largest sample accepted (frames × channels): a minute of 192 kHz stereo.
const MAX_SAMPLE_VALUES: usize = 192_000 * 60 * 2;

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "mix.pad_load",
            "Load Pad Sample",
            [],
            None,
            "{track?, pad: 1-16, path | source: id}: pad `pad` of an instrument track's Drum Pads (made its instrument if it is not already) plays the audio file or session source → {track, pad, source, name, frames}",
            has_selection,
            load
        ),
        cmd!("mix.pad_clear", "Clear Pad Sample", [], None, "{track?, pad: 1-16}: the pad plays nothing", has_selection, clear),
    ]
}

/// The instrument track and 0-based pad a pad command acts on.
fn target(e: &Engine, id: &str, p: &Value) -> Result<(TrackId, usize)> {
    let t = tracks_required(e, id, p)?.first().copied().ok_or_else(|| bad(id, "no track"))?;
    if e.session().track(t).map(|tr| tr.kind) != Some(TrackKind::Instrument) {
        return Err(bad(id, "only instrument tracks have drum pads"));
    }
    let pad = p.get("pad").and_then(Value::as_u64).filter(|n| (1..=PAD_COUNT).contains(n)).ok_or_else(|| bad(id, "`pad` (1-16) required"))?;
    Ok((t, usize::try_from(pad - 1).unwrap_or(0)))
}

fn load(e: &mut Engine, p: &Value) -> Result<Value> {
    let id = "mix.pad_load";
    let (t, pad) = target(e, id, p)?;
    let src = match (p.get("source").and_then(Value::as_u64), str_param(p, "path")) {
        (Some(sid), _) => {
            let sid = SourceId(sid);
            if !e.session().pool.contains(sid) {
                return Err(bad(id, "no such source (or its audio isn't loaded)"));
            }
            sid
        }
        (None, Some(path)) => {
            let path = path.to_string();
            let bytes = crate::io::read_file(&path)?;
            let name = Path::new(&path).file_name().and_then(|n| n.to_str()).unwrap_or("Sample").to_string();
            decode_into_source(e, &name, &bytes, Some(&path))?
        }
        (None, None) => return Err(bad(id, "`path` or `source` required")),
    };
    let info = soundcraft_dsp::plugin_info(PADS).ok_or_else(|| bad(id, "Drum Pads is not available"))?;
    let s = e.session_mut();
    let (frames, name) = s.source(src).map_or((0, String::new()), |x| (x.frames, x.name.clone()));
    let tr = s.track_mut(t).ok_or_else(|| bad(id, "no track"))?;
    if !tr.instrument.as_ref().is_some_and(|i| i.plugin == PADS) {
        let mut ins = Insert::new(PADS);
        for pi in info.params {
            ins.params.insert(pi.id.to_string(), pi.default);
        }
        tr.instrument = Some(ins);
    }
    if let Some(ins) = tr.instrument.as_mut() {
        ins.set_pad(pad, Some(src));
    }
    Ok(json!({"track": t, "pad": pad + 1, "source": src, "name": name, "frames": frames}))
}

/// Decodes `bytes` (a file called `name`, for the format hint) into a new session source at the
/// session rate. Fails before touching the session.
fn decode_into_source(e: &mut Engine, name: &str, bytes: &[u8], path: Option<&str>) -> Result<SourceId> {
    let ext = Path::new(name).extension().and_then(|x| x.to_str()).map(str::to_ascii_lowercase);
    let (format, buf) = soundcraft_audio_io::decode(bytes, ext.as_deref()).map_err(|err| EngineError::Io(format!("{name}: {err}")))?;
    if buf.frames() == 0 || buf.num_channels() == 0 {
        return Err(EngineError::Io(format!("{name}: no audio")));
    }
    let buf = crate::io::at_rate(buf, e.session().sample_rate.hz(), MAX_SAMPLE_VALUES)
        .ok_or_else(|| EngineError::Io(format!("{name}: too long for a pad (a minute of 192 kHz stereo at most)")))?;
    Ok(crate::io::add_source(e.session_mut(), name, buf, path, format.format))
}

fn clear(e: &mut Engine, p: &Value) -> Result<Value> {
    let id = "mix.pad_clear";
    let (t, pad) = target(e, id, p)?;
    let ins = e.session_mut().track_mut(t).and_then(|tr| tr.instrument.as_mut()).filter(|i| i.plugin == PADS);
    let ins = ins.ok_or_else(|| bad(id, "the track's instrument is not Drum Pads"))?;
    let had = ins.pad(pad);
    ins.set_pad(pad, None);
    Ok(json!({"track": t, "pad": pad + 1, "cleared": had}))
}

#[cfg(test)]
mod tests {
    use crate::Engine;
    use serde_json::json;
    use soundcraft_audio_io::{AudioBuffer, BitDepth, EncodeOptions, FileFormat};
    use soundcraft_midi::{Note, Sequence};
    use soundcraft_model::{Clip, TrackId};
    use soundcraft_time::{Range, TICKS_PER_QUARTER};
    use std::path::PathBuf;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sc-pads-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A generated click: `n` frames of `level` (no recorded or third-party audio).
    fn click_wav(path: &std::path::Path, level: f32, n: usize) {
        let buf = AudioBuffer { sample_rate: 48_000, channels: vec![vec![level; n]] };
        let opts = EncodeOptions { format: FileFormat::Wav, bit_depth: BitDepth::Float32, dither: false, bwf: None };
        std::fs::write(path, soundcraft_audio_io::encode(&buf, &opts).unwrap()).unwrap();
    }

    fn instrument_track(e: &mut Engine) -> TrackId {
        e.execute("track.new", &json!({"kind": "instrument", "format": "stereo"})).unwrap();
        e.session().tracks.last().unwrap().id
    }

    /// Hits at (pad note, sixteenth) on the track at 120 bpm.
    fn add_hits(e: &mut Engine, t: TrackId, hits: &[(u8, i64)]) {
        let s16 = TICKS_PER_QUARTER / 4;
        let mut seq = Sequence::default();
        for &(pitch, at) in hits {
            seq.notes.push(Note { pitch, velocity: 127, release_velocity: 64, channel: 0, start: at * s16, length: s16 });
        }
        let s = e.session_mut();
        let cid = s.new_clip_id();
        crate::edit::place_clip(s, t, Clip::midi(cid, "beat", 0, 48_000 * 2, seq));
    }

    #[test]
    fn pads_load_from_files_and_sources_play_on_their_notes_and_undo() {
        let d = dir("play");
        let (kick, snare) = (d.join("kick.wav"), d.join("snare.wav"));
        click_wav(&kick, 0.5, 2000);
        click_wav(&snare, 0.25, 2000);
        let mut e = Engine::default();
        let t = instrument_track(&mut e);
        let r = e.execute("mix.pad_load", &json!({"track": t.0, "pad": 1, "path": kick.to_string_lossy()})).unwrap();
        assert_eq!((r["pad"].as_u64(), r["frames"].as_u64()), (Some(1), Some(2000)));
        e.execute("mix.pad_load", &json!({"track": t.0, "pad": 2, "path": snare.to_string_lossy()})).unwrap();
        // A source already in the session goes on pad 16 without decoding again.
        e.execute("mix.pad_load", &json!({"track": t.0, "pad": 16, "source": r["source"]})).unwrap();
        let ins = e.session().track(t).unwrap().instrument.clone().unwrap();
        assert_eq!(ins.plugin, "drum_pads");
        assert_eq!(ins.pads.len(), 16);
        assert_eq!(ins.pad(15), ins.pad(0));
        // Pad 1 (note 36) on beat 1, pad 2 (note 37) on beat 2: each lands on its sample.
        add_hits(&mut e, t, &[(36, 0), (37, 4)]);
        let out = soundcraft_mix::render_range(e.session(), Range::new(0, 48_000), 512);
        let beat = 24_000;
        assert!(out[0][..2000].iter().all(|x| *x > 0.1), "pad 1 from the first sample");
        assert!(out[0][2000..beat].iter().all(|x| *x == 0.0), "then silence");
        assert!(out[0][beat..beat + 2000].iter().all(|x| *x > 0.05 && *x < out[0][0]), "pad 2, quieter, on beat 2");
        // Clear, and undo all the way back.
        e.execute("mix.pad_clear", &json!({"track": t.0, "pad": 16})).unwrap();
        assert_eq!(e.session().track(t).unwrap().instrument.as_ref().unwrap().pads.len(), 2);
        for _ in 0..4 {
            e.execute("edit.undo", &json!({})).unwrap();
        }
        assert_ne!(e.session().track(t).unwrap().instrument.as_ref().unwrap().plugin, "drum_pads");
        assert!(e.session().sources.is_empty());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn bad_requests_change_nothing() {
        let mut e = Engine::default();
        let t = instrument_track(&mut e);
        e.execute("track.new", &json!({"kind": "audio"})).unwrap();
        let audio = e.session().tracks.last().unwrap().id;
        for p in [
            json!({"track": t.0, "pad": 0, "source": 1}),
            json!({"track": t.0, "pad": 17, "source": 1}),
            json!({"track": t.0, "pad": 1}),
            json!({"track": t.0, "pad": 1, "source": 999}),
            json!({"track": t.0, "pad": 1, "path": "/no/such/file.wav"}),
            json!({"track": audio.0, "pad": 1, "source": 1}),
        ] {
            assert!(e.execute("mix.pad_load", &p).is_err(), "{p}");
        }
        assert!(e.execute("mix.pad_clear", &json!({"track": t.0, "pad": 1})).is_err(), "not Drum Pads yet");
        assert!(e.session().sources.is_empty());
    }

    #[test]
    fn pads_survive_save_and_load() {
        let d = dir("save");
        let wav = d.join("hat.wav");
        click_wav(&wav, 0.5, 500);
        let mut e = Engine::default();
        let t = instrument_track(&mut e);
        e.execute("mix.pad_load", &json!({"track": t.0, "pad": 3, "path": wav.to_string_lossy()})).unwrap();
        let json = e.session().to_json().unwrap();
        let back = soundcraft_model::Session::from_json(&json).unwrap();
        let ins = back.track(t).unwrap().instrument.clone().unwrap();
        assert_eq!(ins.pads, e.session().track(t).unwrap().instrument.as_ref().unwrap().pads);
        assert!(ins.pad(2).is_some() && ins.pad(0).is_none());
        let _ = std::fs::remove_dir_all(d);
    }
}
