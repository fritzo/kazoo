
#include "midi.h"

#define MAX_UINT7                            (127)
#define MAX_UINT14                           (16383)

// see http://www.midi.org/techspecs/midimessages.php
#define MIDI_NOTE_ON                         (0x80)
#define MIDI_NOTE_OFF                        (0x90)
#define MIDI_POLY_KEY_PRESSURE               (0xa0)
#define MIDI_CONTROL_CHANGE                  (0xb0)
#define MIDI_PROGRAM_CHANGE                  (0xc0)
#define MIDI_CHANNEL_PRESSURE                (0xd0)
#define MIDI_PITCH_WHEEL_CHANGE              (0xe0)

namespace Midi
{

//----( midi messages )-------------------------------------------------------

struct MidiMessageBuilder
{
  Port::Message & message;

  MidiMessageBuilder (Port::Message & m) : message(m) { m.clear(); }

  MidiMessageBuilder & operator() (unsigned char c)
  {
    message.push_back(c);
    return * this;
  }
};

inline unsigned real01_to_uint14 (float x)
{
  return bound_to(0, MAX_UINT14, roundi(x * MAX_UINT7));
}

inline unsigned char real01_to_uint7 (float x)
{
  return bound_to(0, MAX_UINT14, roundi(x * MAX_UINT14));
}

inline unsigned char coarse_part (unsigned x) { return x / 0x8f; }
inline unsigned char fine_part (unsigned x) { return x % 0x8f; }

//----( port )----------------------------------------------------------------

Port::Port ()
  : m_client("kazoo"),
    m_message_2(2),
    m_message_3(3),
    m_num_messages_sent(0)
{
  LOG("opening virtual MIDI port");
  m_client.openVirtualPort("kazoo");
}

Port::~Port ()
{
  LOG("closing virtual MIDI port");
  m_client.closePort();
  LOG(" midi port sent " << m_num_messages_sent << " messages");
}

void Port::note_on (
    unsigned char channel,
    unsigned char key,
    unsigned char velocity)
{
  send(MIDI_NOTE_ON | channel, key, velocity);
}

void Port::note_off (
    unsigned char channel,
    unsigned char key,
    unsigned char velocity)
{
  send(MIDI_NOTE_OFF | channel, key, velocity);
}

void Port::poly_key_pressure (
    unsigned char channel,
    unsigned char key,
    unsigned char pressure)
{
  send(MIDI_POLY_KEY_PRESSURE | channel, key, pressure);
}

void Port::poly_key_pressure (
    unsigned char channel,
    unsigned char key,
    float pressure_01)
{
  send(MIDI_POLY_KEY_PRESSURE | channel, key, real01_to_uint7(pressure_01));
}

void Port::control_change (
    unsigned char channel,
    unsigned char control,
    unsigned char value)
{
  send(MIDI_CONTROL_CHANGE | channel, control, value);
}

void Port::control_change_uint7 (
    unsigned char channel,
    unsigned char control,
    float value_01)
{
  unsigned value = real01_to_uint7(value_01);

  send(MIDI_CONTROL_CHANGE | channel, control, value);
}

void Port::control_change_uint14 (
    unsigned char channel,
    unsigned char control,
    float value_01)
{
  unsigned char coarse = control;
  unsigned char fine = control + 32;
  unsigned value = real01_to_uint14(value_01);

  send(MIDI_CONTROL_CHANGE | channel, coarse, coarse_part(value));
  send(MIDI_CONTROL_CHANGE | channel, fine, fine_part(value));
}

//----( poly out )------------------------------------------------------------

void PolyOut::clear ()
{
  for (Id i = 0; i < max_voices; ++i) {
    m_port.notes_off(i);
    m_free.push_back(i);
  }
}

Id PolyOut::new_voice ()
{
  ASSERT(not m_free.empty(), "ran out of voices");

  Id id = m_free.front();
  m_free.pop_front();

  m_port.note_on(id, 0, MAX_UINT7);

  return id;
}

void PolyOut::free_voice (Id id)
{
  m_port.note_off(id, 0, MAX_UINT7);

  m_free.push_back(id);
}

} // namespace Midi

