from pathlib import Path
import re


# Directory configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
TEMP_DIR = ROOT_DIR / "data" / ".temp"


# Display name -> original dataset filename prefix
TARGET_ATTACKS = {
    "DDoS-ICMP": "TCP_IP-DDoS-ICMP",
    "DDoS-SYN": "TCP_IP-DDoS-SYN",
    "DDoS-TCP": "TCP_IP-DDoS-TCP",
    "DDoS-UDP": "TCP_IP-DDoS-UDP",
    "DoS-UDP": "TCP_IP-DoS-UDP",
    "DoS-ICMP": "TCP_IP-DoS-ICMP",
    "DoS-SYN": "TCP_IP-DoS-SYN",
    "DoS-TCP": "TCP_IP-DoS-TCP",
    "MQTT-DDoS-Connect_Flood": "MQTT-DDoS-Connect_Flood",
    "Recon-Port_Scan": "Recon-Port_Scan",
}


def recover_temp():
    """Restore any CSV files left in the temporary directory."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for file in sorted(TEMP_DIR.glob("*.csv")):
        destination = TRAIN_DIR / file.name

        if destination.exists():
            raise FileExistsError(
                f"Cannot restore {file.name}: "
                "destination already exists."
            )

        file.rename(destination)
        print(f"Recovered: {file.name}")


def get_attack_files(prefix):
    """Find every training CSV belonging to an attack type."""
    pattern = re.compile(
        rf"^{re.escape(prefix)}\d*_train\.pcap\.csv$"
    )

    return sorted(
        file
        for file in TRAIN_DIR.glob("*.csv")
        if pattern.fullmatch(file.name)
    )


def get_available_attacks():
    """Associate the ten selected attacks with their CSV files."""
    return {
        name: get_attack_files(prefix)
        for name, prefix in TARGET_ATTACKS.items()
    }


def display_attacks(attacks):
    """Display numbered attack types and their CSV counts."""
    print("\nAvailable Attack Types")
    print("----------------------")

    for number, (name, files) in enumerate(attacks.items(), 1):
        status = f"{len(files)} CSV files"

        if not files:
            status = "NO FILES FOUND"

        print(f"[{number}] {name} ({status})")


def select_attack(attacks):
    """Let the user select an attack type by number."""
    display_attacks(attacks)

    choice = input("\nSelect attack number: ").strip()

    if not choice.isdigit():
        print("Please enter a number.")
        return None

    index = int(choice) - 1
    items = list(attacks.items())

    if not 0 <= index < len(items):
        print("Invalid selection.")
        return None

    attack_name, files = items[index]

    if not files:
        print(f"No CSV files found for {attack_name}.")
        return None

    print(f"\nSelected: {attack_name}")
    print(f"Files to withhold: {len(files)}")

    for file in files:
        print(f"  - {file.name}")

    return attack_name, files


def main():
    recover_temp()

    print("\nIoMT Unseen Attack Detection")
    print("===========================")

    while True:
        print("\n[1] List attack types")
        print("[2] Select attack to withhold")
        print("[0] Exit")

        command = input("\nEnter command: ").strip()

        if command == "0":
            print("Exiting.")
            break

        attacks = get_available_attacks()

        if command == "1":
            display_attacks(attacks)

        elif command == "2":
            select_attack(attacks)

        else:
            print("Unknown command.")


if __name__ == "__main__":
    main()
