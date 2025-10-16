# DAS_Preprocessing

DAS_Preprocessing is a Python toolkit for loading, concatenating, and visualizing **Distributed Acoustic Sensing (DAS)** data.  
It currently supports multiple vendors (OptaSense, Silixa) and includes basic preprocessing tools like detrend, bandpass, f–k filter, and curvelet-like denoise.

---

## Installation

Clone the repository and install the required packages.

```bash
# Using SSH (recommended)
git clone git@github.com:UWGeoD/DAS_Preprocessing.git
cd DAS_Preprocessing

# or using HTTPS
git clone https://github.com/UWGeoD/DAS_Preprocessing.git
cd DAS_Preprocessing

# (optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
