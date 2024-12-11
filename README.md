# propaganda
We are defining the process of fine tuning models to enforce political bias in LLMs from a conservative and liberal parties. 

#DATA
You can see some of the data in the folder data. 
Other data can be scraped from:

- https://partitodemocratico.it/category/comunicati-stampa/
- https://huggingface.co/datasets/Tadorne/amendments
- https://www.allsides.com/unbiased-balanced-news


  #EVAL
  
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
