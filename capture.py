from scapy.all import sniff, IP, TCP, UDP
import csv
from datetime import datetime

output_file = "captured_traffic.csv"

with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "src_ip", "dst_ip", "protocol", "src_port", "dst_port", "packet_size"])

def process_packet(packet):
    if IP in packet:
        proto = "TCP" if TCP in packet else "UDP" if UDP in packet else "OTHER"
        src_port = packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else None)
        dst_port = packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else None)

        row = [
            datetime.now().isoformat(),
            packet[IP].src,
            packet[IP].dst,
            proto,
            src_port,
            dst_port,
            len(packet)
        ]

        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"Captured: {row}")

print("Starting packet capture... Press Ctrl+C to stop.")
sniff(prn=process_packet, store=False)