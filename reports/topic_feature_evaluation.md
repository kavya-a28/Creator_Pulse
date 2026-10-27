# Topic features: comparison

Probe pairs are hand-written diagnostics (16 same-topic, 12 different-topic sharing a keyword), not audience comments. Clustering threshold 0.3.

Multilingual embeddings were not evaluated: not requested (run with --embeddings)

| Features | Same-topic sim | Diff-topic sim | ROC AUC | Same >= thr | Diff < thr | Real: topics | Real: clustered | Real: member sim |
|---|---|---|---|---|---|---|---|---|
| word | 0.355 | 0.201 | 0.542 | 7/16 | 8/12 | 18 | 0.157 | 0.841 |
| char | 0.525 | 0.285 | 0.724 | 11/16 | 5/12 | 40 | 0.338 | 0.839 |
| word+char | 0.44 | 0.243 | 0.641 | 10/16 | 7/12 | 27 | 0.226 | 0.836 |

Per-pair similarity:

| Pair | Kind | Same topic | word | char | word+char |
|---|---|---|---|---|---|
| Python indentation error aa raha hai / How can I fix indentation error in Python? | hinglish vs english | yes | 0.549 | 0.904 | 0.726 |
| Python indentation error aa raha hai / Indentation issue in my Python code | paraphrase | yes | 0.209 | 0.598 | 0.404 |
| How can I fix indentation error in Python? / Indentation issue in my Python code | paraphrase | yes | 0.157 | 0.541 | 0.349 |
| pyhton indentation eror / python indentation error | spelling | yes | 0.124 | 0.531 | 0.327 |
| python ka course bana do please / please make a python course | hinglish vs english | yes | 1.0 | 1.0 | 1.0 |
| java me dsa series laao / DSA in Java please | hinglish vs english | yes | 0.591 | 1.0 | 0.796 |
| stock market course launch karo / please make a trading course | synonym | yes | 0.0 | 0.0 | 0.0 |
| machine learning ka video / ML course please | abbreviation | yes | 0.0 | 0.0 | 0.0 |
| notes ka link kaam nahi kar raha / notes link is not working | hinglish vs english | yes | 0.515 | 0.533 | 0.524 |
| मुझे पायथन सीखनी है / I want to learn python | devanagari vs english | yes | 0.0 | 0.0 | 0.0 |
| react hooks samajh nahi aaya / confused about react hooks | hinglish vs english | yes | 0.531 | 0.494 | 0.513 |
| kali linux ka video dalo / kali linux tutorial please | hinglish vs english | yes | 1.0 | 1.0 | 1.0 |
| system design course please / please make a system design playlist | paraphrase | yes | 1.0 | 1.0 | 1.0 |
| cyber security course / cybersecurity batch kab aayega | spacing variant | yes | 0.0 | 0.555 | 0.277 |
| devops roadmap please / dev ops kaise start kare | spacing variant | yes | 0.0 | 0.243 | 0.122 |
| AI ML course series / artificial intelligence and machine learning course | abbreviation | yes | 0.0 | 0.0 | 0.0 |
| Python indentation error aa raha hai / please make a python course | shared keyword | no | 0.361 | 0.467 | 0.414 |
| java course please / java or python which is better for placements? | shared keyword | no | 0.324 | 0.335 | 0.329 |
| system design course please / game design course please | shared keyword | no | 0.27 | 0.484 | 0.377 |
| AI course launch karo / will AI replace software jobs? | shared keyword | no | 0.325 | 0.212 | 0.268 |
| stock market course / market research job kaise milegi | shared keyword | no | 0.212 | 0.387 | 0.3 |
| data structures course / data science roadmap | shared keyword | no | 0.227 | 0.205 | 0.216 |
| react course please / react native app crash ho raha hai | shared keyword | no | 0.307 | 0.431 | 0.369 |
| how did you get my phone number? / coding ke liye kaun sa phone lu? | shared keyword | no | 0.187 | 0.364 | 0.276 |
| machine learning course / washing machine repair video | shared keyword | no | 0.203 | 0.397 | 0.3 |
| notes ka link do / linkedin profile kaise banaye | unrelated | no | 0.0 | 0.129 | 0.064 |
| dynamic programming lectures / hackathon preparation tips | unrelated | no | 0.0 | 0.015 | 0.007 |
| cloud and devops course / BCA ke baad kya karu | unrelated | no | 0.0 | 0.0 | 0.0 |
