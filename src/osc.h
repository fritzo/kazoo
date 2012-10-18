
#ifndef KAZOO_OSC_H
#define KAZOO_OSC_H

#include "common.h"
#include <oscpack/osc/OscOutboundPacketStream.h>
#include <oscpack/ip/UdpSocket.h>
#include <cstdio> // for sprintf

namespace Osc
{

class UdpOut
{
  enum { buffer_size = 1024 }; // WARNING risk of buffer overflow
  char * const m_buffer;

  osc::OutboundPacketStream m_stream;
  UdpTransmitSocket m_socket;

public:

  UdpOut (const char * address, unsigned port = 7000)
    : m_buffer(new char[buffer_size]),
      m_stream(m_buffer, buffer_size),
      m_socket(IpEndpointName(address, port))
  {}
  ~UdpOut () { delete[] m_buffer; }

  operator osc::OutboundPacketStream & () { return m_stream; }
  void send ()
  {
    m_socket.Send(m_stream.Data(), m_stream.Size());
    m_stream.Clear();
  }
};

class PolyOut
{
  UdpOut m_port;

  char m_address_pattern[32]; // WARNING risk of buffer overflow

public:

  virtual ~PolyOut () {}

  void begin_frame () { m_port << osc::BeginBundleImmediate; }
  void end_frame () { m_port << osc::EndBundle; m_port.send(); }

  void update (size_t id, float x, float y, float z)
  {
    sprintf(m_address_pattern, "kazoo/%i/", id);
    m_port
      << osc::BeginMessage(m_address_pattern)
      << x
      << y
      << z
      << osc::EndMessage;
  }

  void update (size_t id, float x, float y, float z, float u, float v)
  {
    sprintf(m_address_pattern, "kazoo/%i/", id);
    m_port
      << osc::BeginMessage(m_address_pattern)
      << x
      << y
      << z
      << u
      << v
      << osc::EndMessage;
  }
};

} // namespace Osc

#endif // KAZOO_OSC_H

