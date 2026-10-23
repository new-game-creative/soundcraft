//! Live MIDI input (desktop): every MIDI input port, opened with `midir`, sends its note messages
//! to the audio thread, which plays them on the instrument tracks listening to that device (see
//! `AudioState::route_midi`).

use crate::Cmd;
use std::sync::mpsc::Sender;

/// The open MIDI inputs. Dropping it closes them.
pub(crate) struct MidiInputs {
    _conns: Vec<midir::MidiInputConnection<()>>,
    pub(crate) names: Vec<String>,
}

/// Opens every MIDI input port. Ports that fail to open are logged and left out.
pub(crate) fn open(tx: &Sender<Cmd>) -> MidiInputs {
    let mut conns = Vec::new();
    let mut names = Vec::new();
    let ports = match midir::MidiInput::new("Fluid") {
        Ok(m) => m.ports(),
        Err(e) => {
            log::warn!("MIDI input: {e}");
            return MidiInputs { _conns: conns, names };
        }
    };
    for port in ports {
        let Ok(input) = midir::MidiInput::new("Fluid") else { continue };
        let name = input.port_name(&port).unwrap_or_else(|_| "MIDI input".into());
        let Ok(device) = u16::try_from(names.len()) else { break };
        let tx = tx.clone();
        let opened = input.connect(
            &port,
            "fluid-in",
            move |_stamp, msg, ()| {
                if let [status, a, b, ..] = *msg {
                    let _ = tx.send(Cmd::Midi { device, data: [status, a, b] });
                }
            },
            (),
        );
        match opened {
            Ok(c) => {
                conns.push(c);
                names.push(name);
            }
            Err(e) => log::warn!("MIDI input {name}: {e}"),
        }
    }
    MidiInputs { _conns: conns, names }
}
