---
title: Make Trig Times
has_children: false
parent: ECoG In Unit
---
Command line: edit maketrigtimes.m

Run each section 

Guideline for seconds_between_triggers for tasks

| **Task**                          | **Seconds_between_Triggers** |
|-----------------------------------|------------------------------|
| Lexical Within 2X Delay Speak     | 2.5                          |
| Lexical Within 2X No Delay Button | 1.75                         |
| Sentence Rep                      | 2.0                          |
| Phoneme Sequencing                | 1.0                          |
| Uniqueness Point                  | 2.0                          |
| Neighborhood_Sternberg            | .75                          |
| Environmental_Sternberg           | .75                          |
| Timit                             |                              |


*This setting will label every trigger. In the identify triggers section add in this line to pick the alternate triggers:

trigTimes = trigTimes(1:2:end);

trigTimes should be equal to trialInfo 

Number of triggers for each task:

<span style="font-family:Calibri, sans-serif;">Timit- 336</span>

<span style="font-family:Calibri, sans-serif;">Delay- 336</span>

<span style="font-family:Calibri, sans-serif;">No Delay- 504</span>

<span style="font-family:Calibri, sans-serif;">Uniqueness Point- 480</span>

<span style="font-family:Calibri, sans-serif;">Sentence Rep- 270</span>

<span style="font-family:Calibri, sans-serif;">PhonemeS- 208</span>

<span style="font-family:Calibri, sans-serif;">Environmental- 1176</span>

<span style="font-family:Calibri, sans-serif;">Neighborhood- 1120</span>