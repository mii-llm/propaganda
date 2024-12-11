# Political Compass

A simple tool to determine the italian political alignment of LLM.

## Installation

```bash
pip install italian-political-compass

```python

from italian_political_compass import ItalianPoliticalCompass

compass = ItalianPoliticalCompass()
results = compass.calculate_alignment()
compass.print_results(verbose=False)

#or

compass = ItalianPoliticalCompass(model_name="different/model-name")
results = compass.calculate_alignment()
compass.print_results(verbose=True) # prints questions and responses

# and

compass = ItalianPoliticalCompass()
parties = compass.get_supported_parties()
print(parties)


