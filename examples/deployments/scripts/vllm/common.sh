#!/bin/bash

load_config() {
    local config_file="${CONFIG_FILE:-$(dirname "${BASH_SOURCE[0]}")/config.properties}"
    
    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config file '$config_file' not found!" >&2
        exit 1
    fi

    while IFS= read -r line; do
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$line" || "$line" == \#* ]] && continue

        if [[ "$line" == export\ * ]]; then
            rest="${line#export }"
            eval "export $rest"
        else
            if [[ "$line" == *=* ]]; then
                key="${line%%=*}"
                value="${line#*=}"
                key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                eval "$key=\$value"
            else
                echo "WARNING: Invalid config line (no '=' found): $line" >&2
            fi
        fi
    done < "$config_file"
}

ensure_ifconfig_installed() {
    if command -v ifconfig >/dev/null 2>&1; then
        return 0
    fi

    echo "'ifconfig' not found. Attempting to install net-tools..."

    if command -v apt-get >/dev/null 2>&1; then
        echo "Detected apt-get (Debian/Ubuntu). Installing net-tools..."
        sudo apt-get update && sudo apt-get install -y net-tools
    elif command -v yum >/dev/null 2>&1; then
        echo "Detected yum (RHEL/CentOS). Installing net-tools..."
        sudo yum install -y net-tools
    elif command -v dnf >/dev/null 2>&1; then
        echo "Detected dnf (Fedora). Installing net-tools..."
        sudo dnf install -y net-tools
    else
        echo "ERROR: No supported package manager (apt/yum/dnf) found."
        echo "Please install 'net-tools' manually, 'ifconfig' is required to get network interface information."
        exit 1
    fi

    if ! command -v ifconfig >/dev/null 2>&1; then
        echo "ERROR: Failed to install net-tools. Please install 'net-tools' manually, 'ifconfig' is required to get network interface information."
        exit 1
    fi

    echo "✅ ifconfig is now available."
}

get_interface_by_ip() {
    local target_ip="$1"
    ifconfig | awk -v target="$target_ip" '
        /^[[:alnum:]]/ {
            iface = $1
            sub(/:$/, "", iface)  
        }
        /inet / {
            for (i = 1; i <= NF; i++) {
                gsub(/addr:/, "", $i)
                if ($i == target) {
                    print iface
                    exit
                }
            }
        }
    '
}