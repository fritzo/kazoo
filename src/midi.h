
#ifndef KAZOO_MIDI_H
#define KAZOO_MIDI_H

/** A simple C++ wrapper object for portmidi streams
*/

#include "common.h"
#include "gestures.h"
#include "vectors.h"
#include <stk/RtMidi.h>
#include <deque>
#include "hash_map.h"

/** Virtual Midi Controller.

  This uses the RtMidi component http://www.music.mcgill.ca/~gary/rtmidi
  of the Synthesis ToolKit library https://ccrma.stanford.edu/software/stk

  Although the libaries are cross-platform (Mac OS X, Linux, Irix, Windows),
  the virtual midi port functionality is only available for Mac OS X and Linux.

  Errors are thrown and not caught here.
  If errors become a problem, the source can be compiled with __RTMIDI_DEBUG__,
  or added to the kazoo project via the three files
    RtMidi.h
    RtError.h
    RtMidi.cpp
  with appropriate platform-dependent compile flags.

  The Licence for both libraries is BSD-style, and request:
  "If you make a million dollars with it, it would be nice if you would share"

  References:
  * Free midi spec as old-school html
    http://www.blitter.com/~russtopia/MIDI/~jglatt/tech/midispec.htm
  * Midi Continuous controllers page
    http://253.ccarh.org/handout/controllers/
  * midi message specs:
    http://www.midi.org/techspecs/midimessages.php
  * how to get software / hardware synthesis set up in linux:
    https://help.ubuntu.com/community/Midi
*/

namespace Midi
{

//----( port )----------------------------------------------------------------

class Port
{
  RtMidiOut m_client;

public:

  typedef std::vector<unsigned char> Message;

private:

  Message m_message_2;
  Message m_message_3;

  size_t m_num_messages_sent;

public:

  Port ();
  ~Port ();

  // general message
  void send (Message & message)
  {
    m_client.sendMessage(& message);
    ++m_num_messages_sent;
  }

  void send (unsigned char x, unsigned char y)
  {
    m_message_2[0] = x;
    m_message_2[1] = y;
    send(m_message_2);
  }

  void send (unsigned char x, unsigned char y, unsigned char z)
  {
    m_message_3[0] = x;
    m_message_3[1] = y;
    m_message_3[2] = z;
    send(m_message_3);
  }

  // specific messages
  void note_on (
      unsigned char channel,
      unsigned char key,
      unsigned char velocity);

  void note_off (
      unsigned char channel,
      unsigned char key,
      unsigned char velocity);

  void poly_key_pressure (
      unsigned char channel,
      unsigned char key,
      unsigned char pressure);

  void poly_key_pressure (
      unsigned char channel,
      unsigned char key,
      float pressure_01);

  void control_change (
      unsigned char channel,
      unsigned char control,
      unsigned char value);

  void control_change_uint7 (
      unsigned char channel,
      unsigned char control,
      float value_01);

  void control_change_uint14 (
      unsigned char channel,
      unsigned char control,
      float value_01);

  void sound_off (unsigned char channel) { control_change(channel, 120, 0); }
  void notes_off (unsigned char channel) { control_change(channel, 123, 0); }
};

//----( poly out )------------------------------------------------------------

/** Multi-channel chorus
 *
 * This implementation maps up to 16 voices of vector data to
 * the first key in each of 16 channels of a midi port,
 * with parameters optionally mapped to
 *   x = 14-bit effect control 1 = continuous control 12
 *   y = 14-bit effect control 2 = continuous control 13
 *   z = 14-bit breath control   = continuous control 2
 *   energy = 14-bit main volume
 *   ...
 */

class PolyOut
{
  enum {
    max_voices = 16,
    z_control = 7,  // main volume
    x_control = 12, // effect_control 1
    y_control = 13, // effect_control 2
    u_control = 14,
    v_control = 15
  };

  Port m_port;

  std::deque<Id> m_free;

public:

  void clear ();

  PolyOut () { clear(); }
  virtual ~PolyOut () { clear(); }

  Id new_voice ();
  void free_voice (Id id);

  void update_voice (Id id, float x, float y, float z)
  {
    m_port.control_change_uint14(id, x_control, x);
    m_port.control_change_uint14(id, y_control, y);
    m_port.control_change_uint14(id, z_control, z);
  }

  void update_voice (Id id, float x, float y, float z, float u, float v)
  {
    m_port.control_change_uint14(id, x_control, x);
    m_port.control_change_uint14(id, y_control, y);
    m_port.control_change_uint14(id, z_control, z);
    m_port.control_change_uint14(id, u_control, u);
    m_port.control_change_uint14(id, v_control, v);
  }

  void update_voice (Id id, const Gestures::Finger & finger)
  {
    // TODO add energy control
    // TODO add velocity control
    m_port.control_change_uint14(id, x_control, finger.get_x());
    m_port.control_change_uint14(id, y_control, finger.get_y());
    m_port.control_change_uint14(id, z_control, finger.get_z());
  }
};

//----( chorus )--------------------------------------------------------------

/** A midi chorus controller, inspired by Synthesis::Chorus
 */
template<class Descriptor>
class Chorus
{
  typedef std::hash_map<Id, Id> Voices;
  Voices m_voices;

public:

  Chorus () {}
  virtual ~Chorus () { clear(); }

  virtual void sample (
      Vector<Id> & ids,
      Vector<Descriptor> & descriptors,
      PolyOut & poly_out);

  void clear ();
};

template<class Descriptor>
void Chorus<Descriptor>::sample (
    Vector<Id> & ids,
    Vector<Descriptor> & descriptors,
    PolyOut & poly_out)
{
  Voices ended;
  std::swap(m_voices, ended);

  // sample continuing voices & turn on new voices
  typedef typename Voices::iterator Auto;
  for (size_t i = 0; i < ids.size; ++i) {
    Id id = ids[i];
    const Descriptor & descriptor = descriptors[i];

    Auto v = ended.find(id);
    if (v != ended.end()) {
      Id midi_id = v->second;

      poly_out.update_voice(midi_id, descriptor);

      m_voices.insert(*v);
      ended.erase(v);
    } else {
      Id midi_id = poly_out.new_voice();

      poly_out.update_voice(midi_id, descriptor);

      m_voices.insert(typename Voices::value_type(id, midi_id));
    }
  }

  // turn off ended voices
  for (Auto v = ended.begin(); v != ended.end(); ++v) {
    Id midi_id = v->second;

    poly_out.free_voice(midi_id);
  }
}

template<class Descriptor>
void Chorus<Descriptor>::clear ()
{
  m_voices.clear();
}

} // namespace Midi

#endif // KAZOO_MIDI_H

