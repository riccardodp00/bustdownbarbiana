"""
Main entrypoint for the project. It imports the pipeline func from pipeline.py and
handles the thematic alignment analysis of a journal of one's choosing taken from 
PubMed's archive. The journal can be set in config.py. 

The current journal of choice is Forensic Science International.

"""

from src.pipeline import pipeline

def main():
  pipeline()

if __name__ == "__main__":
  main()