"""
Loads environmental variables stored in .env through dotenv.
Stores all user-tunable constants (the configuration).

"""

from __future__ import annotations
import os
from dotenv import load_dotenv

# load environment variables from local .env file
load_dotenv()

# defining the workspace
DIRECTORY_DATA = "data"
DIRECTORY_GRAPHS = "graphs"

# journal data
SOURCE = "pubmed"
JOURNAL = "Forensic Science International"
START = 2015
END = 2023
AIMS_AND_SCOPE = """
    An international journal dedicated to the applications of medicine and science in the administration of justice.
    Forensic Science International is the flagship journal in the prestigious Forensic Science International family,
    publishing the most innovative, cutting-edge, and influential contributions across the forensic sciences.
    Fields include: forensic pathology and histochemistry, chemistry, biochemistry and toxicology, biology, serology, odontology, psychiatry,
    anthropology, digital forensics, the physical sciences, firearms, and document examination,
    as well as investigations of value to public health in its broadest sense, and the important marginal area where science and medicine interact with the law.
    
"""

AIMS_AND_SCOPE_URL = "https://www.sciencedirect.com/journal/forensic-science-international"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# technical data
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 32
PUBMED_RETMAX_DEFAULT = 5000
PUBMED_EFETCH_BATCH_SIZE = 200
PUBMED_THROTTLE_SECONDS = 0.12  # increase if no API key / you hit rate limits
N_TOPICS = 20

# enviromental variables
PM_EMAIL = os.getenv("PUBMED_EMAIL", "")
PM_API_KEY = os.getenv("PUBMED_API_KEY", None)
HF_EMAIL = os.getenv("HF_EMAIL", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)