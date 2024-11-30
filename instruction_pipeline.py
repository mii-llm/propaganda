from rich import print
from openai import AsyncClient
from dotenv import load_dotenv
import os
from asyncio import gather, sleep, run
import json
from typing import List, Dict
from pathlib import Path
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import trange

SYSTEM_POMPT_QUESTION = """Sei un assistente AI che utilizza un approccio Chain of Thought (CoT) with reflection per generare una domanda diretta e contestualizzata a partire da una dichiarazione ufficiale di un partito politico.

## Obbiettivo

Il tuo obbiettivo è quello di:

  - Leggere attentamente la dichiarazione identificando il tema principale e il messaggio chiave che il partito vuole comunicare.
  - Formulare una domanda che un cittadino potrebbe porre per ottenere maggiori informazioni o chiarimenti. La domanda deve essere:
    - Chiara e contestualizzata, evitando termini tecnici o complessi.
    - Informale, come se fosse posta in una conversazione quotidiana.
    - Genuina, mostrando un interesse reale verso la questione.
    - Diretta al punto, senza ambiguità o giri di parole.
  - Assicurati che la risposta naturale alla domanda sia la dichiarazione stessa o una sua parte significativa. Se la domanda non porta direttamente alla dichiarazione, riformula la domanda in modo che sia più pertinente.


## Approccio step-by-step

Per generare la domanda segui attentamente i seguenti passaggi:

1. Ragiona sul problema passo dopo passo all'interno dei tag <thinking>.
2. Rifletti sul tuo ragionamento per verificare eventuali errori o miglioramenti all'interno dei tag <reflection>.
3. Apporta le modifiche necessarie in base alla tua riflessione.
4. Fornisci la tua risposta finale e concisa all'interno dei tag <output>.

Importante: Le sezioni <thinking> e <reflection> sono esclusivamente dedicate al tuo processo di ragionamento interno. Non includere parti della tua risposta finale in queste sezioni. La risposta effettiva alla domanda deve essere contenuta interamente nei tag <output>.

Usa il seguente formato di output per la tua risposta:

```xml
<thinking>
    [Il tuo ragionamento step-by-step va qui. Questo è il tuo processo di pensiero interno, non la risposta finale]
<reflection>
    [Riflessione sul tuo ragionamento, per verificare eventuali errori o miglioramenti]
</reflection>
    [Eventuali modifiche al tuo pensiero basate sulla tua riflessione]
</thinking>
<output>
    [La tua domanda definitiva generata. Questa è l'unica parte che verrà mostrata all'utente]
</output>
```"""

SYSTEM_PROMPT_ANSWER = """Sei un assistente AI progettato per rispondere a domande basandoti esclusivamente su una dichiarazione politica fornita dall'utente. Rispondi utilizzando lo stile linguistico e la retorica tipici del politico autore della dichiarazione.

## Linee guida per la generazione della risposta

Per generare la risposta segui attentamente le regole a venire:

1. Esprimi le idee della dichiarazione in modo diretto, assertivo e sicuro, evitando costruzioni che implicano appelli diretti o richieste (es. 'chiediamo che il governo prenda' -> 'il governo deve prendere'). Formula affermazioni forti e indipendenti.
2. Non fare riferimento a partiti, movimenti o entità collettive. Comunica il messaggio come un insieme di principi e convinzioni.
3. Usa esclusivamente le informazioni contenute nella dichiarazione per rispondere, senza introdurre opinioni personali, aggiunte o contraddizioni.
4. Adotta lo stile e il tono del politico, ma mantieni sempre il ruolo di un assistente AI che articola risposte convincenti e incisive.
5. Mantieni il linguaggio chiaro, persuasivo e orientato a trasmettere le idee centrali della dichiarazione senza ambiguità o inutili formalità.

Il tuo obiettivo è riflettere fedelmente il messaggio della dichiarazione, rispondendo con precisione, autenticità e autorevolezza."""

PROMPT_TEMPLATE = """## Dichiarazione politica

{statement}

## Domanda

{question}"""

load_dotenv()

@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(multiplier=0.5, max=60))
async def generate(client: AsyncClient, system: str, prompt: str):
    response = await client.chat.completions.create(
        model="llama3.1-70b",
        n=1,
        messages=[
            {
                'role': 'system',
                'content': system
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        temperature=0.2,
        top_p=1,
        max_completion_tokens=8192
    )

    return response.choices[0].message.content

def parse_questions(questions: List[str]):
    parsed_question = []
    for question in questions:
        output = re.search(r"<output>\s*(.*?)\s*</output>", question, re.DOTALL)
        output = output.group(1) if output else ""
        parsed_question.append({'reasoning': question, 'question': output})

    return parsed_question

def parse_answers(responses: List[str]):
    parsed_responses = []
    for response in responses:
        answer = re.sub(r'\"', r'', response, 0, re.MULTILINE)
        parsed_responses.append(answer)

    return parsed_responses

async def piepline(
    client: AsyncClient,
    data: List[Dict],
    batch_size: int,
    output_file: Path
):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'a') as fp:
        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            questions = await gather(*[
                generate(
                    client=client,
                    system=SYSTEM_POMPT_QUESTION,
                    prompt=doc['text']
                ) for doc in batch
            ])
            parsed_questions = parse_questions(questions)

            responses = await gather(*[
                generate(
                    client=client,
                    system=SYSTEM_PROMPT_ANSWER,
                    prompt=PROMPT_TEMPLATE.format(
                        statement=doc['text'],
                        question=question['question']
                    )
                ) for doc, question in zip(batch, parsed_questions)
            ])

            parsed_answers = parse_answers(responses)

            for doc, question, answer in zip(batch, parsed_questions, parsed_answers):
                fp.write(json.dumps({
                    'group': doc['group'],
                    'date': doc['date'],
                    'statement': doc['text'],
                    **question,
                    'answer': answer
                }) + '\n')

            fp.flush()

            await sleep(2)


if __name__ == '__main__':
    client = AsyncClient(
        api_key=os.environ['CEREBRAS_API_KEY'],
        base_url="https://api.cerebras.ai/v1"
    )

    with open('./data/comunicati_parz/archive.json', 'r' ) as fp:
        data = json.load(fp)

    data = data[310:]

    run(piepline(
        client=client,
        data=data,
        batch_size=5,
        output_file=Path('./output/instructions.jsonl')
    ))
