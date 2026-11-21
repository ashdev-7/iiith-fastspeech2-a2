# Leakage control: emotion2vec+ seed's own head on TESS (post hoc, descriptive)

8 random clips per speaker x emotion (112), no training on our side; unrestricted top label.

|                    |   angry |   disgusted |   fearful |   happy |   neutral |   other |   sad |
|:-------------------|--------:|------------:|----------:|--------:|----------:|--------:|------:|
| ('OAF', 'angry')   |       8 |           0 |         0 |       0 |         0 |       0 |     0 |
| ('OAF', 'disgust') |       0 |           8 |         0 |       0 |         0 |       0 |     0 |
| ('OAF', 'fear')    |       0 |           0 |         8 |       0 |         0 |       0 |     0 |
| ('OAF', 'happy')   |       0 |           0 |         0 |       8 |         0 |       0 |     0 |
| ('OAF', 'neutral') |       0 |           0 |         0 |       0 |         8 |       0 |     0 |
| ('OAF', 'ps')      |       0 |           0 |         0 |       0 |         0 |       8 |     0 |
| ('OAF', 'sad')     |       0 |           0 |         0 |       0 |         0 |       0 |     8 |
| ('YAF', 'angry')   |       8 |           0 |         0 |       0 |         0 |       0 |     0 |
| ('YAF', 'disgust') |       0 |           8 |         0 |       0 |         0 |       0 |     0 |
| ('YAF', 'fear')    |       0 |           0 |         8 |       0 |         0 |       0 |     0 |
| ('YAF', 'happy')   |       0 |           0 |         0 |       8 |         0 |       0 |     0 |
| ('YAF', 'neutral') |       0 |           0 |         0 |       0 |         8 |       0 |     0 |
| ('YAF', 'ps')      |       0 |           0 |         0 |       0 |         0 |       8 |     0 |
| ('YAF', 'sad')     |       0 |           0 |         0 |       0 |         0 |       0 |     8 |
