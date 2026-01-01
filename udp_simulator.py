import argparse
import asyncio
from datetime import datetime
import socket

SAMPLE_RADIOINFO = """
<RadioInfo>
  <RadioNr>1</RadioNr>
  <Freq>352211</Freq>
  <TXFreq>352211</TXFreq>
  <Mode>CW</Mode>
  <IsRunning>True</IsRunning>
  <IsTransmitting>False</IsTransmitting>
  <FocusRadioNr>1</FocusRadioNr>
  <ActiveRadioNr>1</ActiveRadioNr>
</RadioInfo>
"""

SAMPLE_CONTACT = """
<contactinfo>
  <timestamp>{ts}</timestamp>
  <band>3.5</band>
  <rxfreq>352519</rxfreq>
  <call>W1AW</call>
  <wpxprefix>W1</wpxprefix>
  <countryprefix>K</countryprefix>
  <continent>NA</continent>
  <points>1</points>
  <ismultiplierl>1</ismultiplierl>
  <radionr>1</radionr>
  <ID>QSO-1</ID>
</contactinfo>
"""

SAMPLE_SPOT = """
<spot>
  <dxcall>AL3CDE</dxcall>
  <frequency>7061.2</frequency>
  <spottercall>K2PO/7-#</spottercall>
  <timestamp>{ts}</timestamp>
  <action>add</action>
  <mode>CW</mode>
  <comment>CW 9 DB 18 WPM CQ AK</comment>
  <status>single mult</status>
  <statuslist>single mult</statuslist>
</spot>
"""


def send_packet(host: str, port: int, payload: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(payload.encode(), (host, port))
    sock.close()


def main():
    parser = argparse.ArgumentParser(description="UDP simulator for AI Contest Assistant")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12060)
    parser.add_argument("--mode", choices=["radio", "contact", "spot", "all"], default="all")
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if args.mode in {"radio", "all"}:
        send_packet(args.host, args.port, SAMPLE_RADIOINFO)
        print("Sent RadioInfo")
    if args.mode in {"contact", "all"}:
        send_packet(args.host, args.port, SAMPLE_CONTACT.format(ts=ts))
        print("Sent ContactInfo")
    if args.mode in {"spot", "all"}:
        send_packet(args.host, args.port, SAMPLE_SPOT.format(ts=ts))
        print("Sent Spot")


if __name__ == "__main__":
    main()
