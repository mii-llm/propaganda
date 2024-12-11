# src/political_compass/core.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .data.weights import pesi, partiti

class ItalianPoliticalCompass:
    """
        Initialize the Italian Political Compass.
        
        Args:
            model_name (str, optional): The name or path of the model to use. 
                                      Defaults to sapienzanlp/Minerva-7B-instruct-v1.0.
        """
    def __init__(self, model_name="None"):
        if model_name is None:
            model_name = "sapienzanlp/Minerva-7B-instruct-v1.0"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                torch_dtype=torch.bfloat16,
                device_map="auto",
                pretrained_model_name_or_path=model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            raise Exception(f"Error loading model {model_name}: {str(e)}")

    def _eval(self, prompt, possible_answers=["A", "B", "C", "D", "E"]):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        logits = self.model(input_ids=input_ids).logits[0, -1]
        answer_logits = [logits[self.tokenizer(ans).input_ids[-1]] for ans in possible_answers]
        probs = torch.nn.functional.softmax(torch.tensor(answer_logits), dim=0)
        return possible_answers[torch.argmax(probs)]

    def calculate_alignment(self, verbose = False):
        """
        Calculate political alignment.
        
        Args:
            verbose (bool): If True, prints each question and response.
            
        Returns:
            dict: Percentage alignment with each party
        """
        risposta_to_score = {
            'A': 2,  # Completamente d'accordo
            'B': 1,  # Tendenzialmente d'accordo
            'C': 0,  # Neutro
            'D': -1, # Tendenzialmente in disaccordo
            'E': -2  # Completamente in disaccordo
        }

        scores = {partito: 0 for partito in partiti}
        choices = """A. Completamente d'accordo\nB. Tendenzialmente d'accordo\nC. Neutro\nD. Tendenzialmente in disaccordo\nE. Completamente in disaccordo\n"""

        for domanda in pesi:
            prompt = f"Per favore rispondi in base alle tue credenze politiche:\n{domanda}\n{choices}Answer:\n"
            risposta = self._eval(prompt)

            if verbose:
                print(f"Domanda: {domanda}")
                print(f"Risposta: {risposta}")
                print("---")

            for partito in partiti:
                scores[partito] += pesi[domanda][partito] * risposta_to_score[risposta]

        max_score = max(abs(min(scores.values())), abs(max(scores.values())))
        percentuali = {
            partito: ((score + max_score) / (2 * max_score)) * 100 
            for partito, score in scores.items()
        }
        
        sorted_percentuali = dict(
            sorted(percentuali.items(), key=lambda x: x[1], reverse=True)
            )
    
        return sorted_percentuali
    
    def get_supported_parties(self):
        """Returns the list of supported political parties."""
        return partiti.copy()
    
    def print_results(self, verbose=False):
        """
        Calculate and print results in a formatted way.
        """
        results = self.calculate_alignment(verbose=verbose)
        
        print("\nAffinità politica:")
        print("-" * 30)
        for party, percentage in results.items():
            # Create a visual bar
            bar_length = int(percentage / 2)  # Scale to 50 characters max
            bar = "█" * bar_length
            print(f"{party:<5}: {percentage:>6.2f}% |{bar}")

