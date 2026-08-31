from pathlib import Path
import sys
PROJECT=Path("/content/drive/MyDrive/Clinderma_Pigmentation")
sys.path.insert(0,str(PROJECT))
%run "$PROJECT/src/data/dataset_discovery_download.py"
