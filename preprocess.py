import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

df = pd.read_csv("KDDTrain+.txt", names=columns)
df = df.dropna()

le = LabelEncoder()
df["protocol_type"] = le.fit_transform(df["protocol_type"])
df["service"] = le.fit_transform(df["service"])
df["flag"] = le.fit_transform(df["flag"])

attack_map = {
    "normal": "Normal",
    "neptune": "DoS", "smurf": "DoS", "back": "DoS", "teardrop": "DoS",
    "pod": "DoS", "land": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    "satan": "PortScan", "ipsweep": "PortScan", "nmap": "PortScan",
    "portsweep": "PortScan", "mscan": "PortScan", "saint": "PortScan",
    "guess_passwd": "BruteForce", "ftp_write": "BruteForce",
    "imap": "BruteForce", "multihop": "BruteForce", "phf": "BruteForce",
    "warezmaster": "BruteForce", "warezclient": "BruteForce",
    "spy": "BruteForce", "xlock": "BruteForce", "xsnoop": "BruteForce",
    "snmpguess": "BruteForce", "snmpgetattack": "BruteForce",
    "httptunnel": "BruteForce", "sendmail": "BruteForce", "named": "BruteForce",
    "buffer_overflow": "DataExfiltration", "loadmodule": "DataExfiltration",
    "rootkit": "DataExfiltration", "perl": "DataExfiltration",
    "sqlattack": "DataExfiltration", "xterm": "DataExfiltration",
    "ps": "DataExfiltration", "worm": "DataExfiltration"
}

df["label"] = df["label"].map(attack_map).fillna("Other")

X = df.drop(["label", "difficulty"], axis=1)
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing done!")
print("Label counts:\n", y.value_counts())

import joblib
joblib.dump(scaler, "scaler.pkl")
X.to_csv("X_processed.csv", index=False)
y.to_csv("y_processed.csv", index=False)
print("Saved: scaler.pkl, X_processed.csv, y_processed.csv")