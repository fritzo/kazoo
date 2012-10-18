
#include "gestures.h"

#define TOL (1e-8f)

namespace Gestures
{

//----( gesture recognition )-------------------------------------------------

inline Id chord_id (Id x, Id y)
{
  const size_t shift = sizeof(Id) / 2;
  return x ^ (y >> shift) ^ (y << shift);
}

void fingers_to_chords (
    const BoundedMap<Id, Finger> & fingers,
    BoundedMap<Id, Chord> & chords,
    float chord_width,
    float chord_height)
{
  chords.clear();
  for (size_t i = 0; i < fingers.size; ++i) { Id id0 = fingers.keys[i];
  for (size_t j = 0; j < fingers.size; ++j) { Id id1 = fingers.keys[j];
    if (chords.full()) return;
    if (id0 >= id1) continue;

    const Finger & f0 = fingers.values[i];
    const Finger & f1 = fingers.values[j];
    bool close = f0.distance_to(f1, chord_width, chord_height) < 1;
    if (not close) continue;

    Id id = chord_id(id0, id1);
    chords.add(id).set(f0, f1, chord_width, chord_height);
  }}
}

void relativize_fingers (
    BoundedMap<Id, Finger> & hands,
    BoundedMap<Id, Finger> & fingers,
    float sigma_x,
    float sigma_y)
{
  // shift hands up to finger position, assuming upright hands
  for (size_t j = 0; j < hands.size; ++j) {
    Finger & hand = hands.values[j];
    hand.y() += sigma_y;
  }

  // shift fingers relative to expected hand position
  for (size_t i = 0; i < fingers.size; ++i) {
    Finger & finger = fingers.values[i];

    float total_assoc = 0.0f;
    for (size_t j = 0; j < hands.size; ++j) {
      Finger & hand = hands.values[j];

      float dist = hand.distance_to(finger, sigma_x, sigma_y);
      float assoc = hand.z() * exp(-sqr(dist) / 2);
      total_assoc += assoc;
    }

    finger.z() *= total_assoc;
    finger.z_t() *= total_assoc;

    if (total_assoc < TOL) continue;
    float assoc_scale = 1.0f / total_assoc;

    for (size_t j = 0; j < hands.size; ++j) {
      Finger & hand = hands.values[j];

      float dist = hand.distance_to(finger, sigma_x, sigma_y);
      float assoc = hand.z() * exp(-sqr(dist) / 2);
      float part = assoc * assoc_scale;

      finger.x() -= part * hand.x();
      finger.y() -= part * hand.y();
      finger.x_t() -= part * hand.x_t();
      finger.y_t() -= part * hand.y_t();
    }
  }

  // shift hands back down
  for (size_t j = 0; j < hands.size; ++j) {
    Finger & hand = hands.values[j];
    hand.y() -= sigma_y;
  }
}

}

