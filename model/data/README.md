# Dataset Detail

## Food Ordering Ontology

1. name.person : Name of person [0 data]
1. num.people :  Number of person

1. type.food :  
1. type.meal :

1. name.item :
1. other_description.item :


1. total_price :
1. price_range :
1. time.pickup :

1. name.restaurant :
1. location.restaurant :
1. rating.restaurant :
1. phone.restaurant :
1. business_hours.restaurant : 
1. official_description.restaurant :

1. type.retrieval :

## STRUCTURE

Each conversation in the data file has the following structure:

- __conversation_id:__ A universally unique identifier with the prefix 'dlg-'. The ID has no meaning.
- __utterances:__ An array of utterances that make up the conversation.
- __instruction_id:__ A reference to the file(s) containing the user (and, if applicable, agent) instructions for this conversation.


Each utterance has the following fields:

- __index:__ A 0-based index indicating the order of the utterances in the conversation.
- __speaker:__ Either USER or ASSISTANT, indicating which role generated this utterance.
- __text:__ The raw text of the utterance. 'ASSISTANT' turns are originally written (then played to the user via TTS) and 'USER' turns are transcribed from the spoken recordings of crowdsourced workers.
- __segments:__ An array of various text spans with semantic annotations.


Each segment has the following fields:

- __start_index:__ The position of the start of the annotation in the utterance text.
- __end_index:__ The position of the end of the annotation in the utterance text.
- __text:__ The raw text that has been annotated.
- __annotations:__ An array of annotation details for this segment.


Each annotation has a single field:

- __name:__ The annotation name.