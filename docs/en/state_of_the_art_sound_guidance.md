# State of the art: sound for guiding blind and visually impaired people

> English translation of the team's [`docs/etat_de_lart_guidage_sonore.md`](../etat_de_lart_guidage_sonore.md). The French original is the reference. Source titles, links and ✅/📚 markers are kept as in the original.

*Research phase (22/11/2026). Nothing is coded yet. This document is a basis for finding innovative leads for VisionAid.*

> **Reliability of sources.** References marked ✅ were found online during this research (usually the abstract or a summary, not always the full text). Those marked 📚 come from the classic literature of the field and were not re-checked: details (figures, year) must be verified before citing them in a report.

---

## 0. In short (the 10 lessons to remember)

1. **A spatialised sound requires less effort than a spoken instruction** when the user is already busy (crossing, conversation, memory). Without cognitive load, both are equal. This is the most solid result in the field (Klatzky, Loomis et al. 2006).
2. **Voice remains better for meaning** ("door", "stairs", "arrived") and for learning quickly. Sounds are better for direction and continuous adjustment. The systems that work are **hybrid**.
3. **Preferences vary a lot between people and with their expertise.** The amount of information must be adjustable (Ahmetovic et al. 2019).
4. **A broadband sound (noise, click, sound rich in harmonics) is localised much better than a pure sine wave.** "Sonar"-type beaconing gave the best results in Walker & Lindsay (2006).
5. **A capture radius of about 1.5 m around a waypoint** is a good compromise. Too small, the user overshoots the point then searches for it; too large, they cut corners.
6. **Head tracking matters more than a personalised HRTF.** It eliminates almost all front/back confusions.
7. **The headset must not block the ears.** Blind people use traffic noise to orient themselves and to cross. Bone conduction, open earphones and transparency mode are the norm; bone conduction localises almost as well as a classic headset.
8. **Tempo is the most effective way to signal urgency**, ahead of pitch (Edworthy, Hellier). The "parking sensor beep" is a mental model everyone understands.
9. **A continuous sound is tiring and ends up being ignored.** The best systems stay silent when everything is fine and only signal deviations.
10. **"8D audio" is not a scientific technique.** It is a marketing effect (rotating panning + reverb). What matters is HRTF binaural, ambisonics and parametric rendering.

---

## 1. Voice or sounds: what the studies say

### 1.1 Cognitive load: the advantage of spatialised sound

- ✅ **Klatzky, Marston, Giudice, Golledge & Loomis (2006)**, *Cognitive load of navigating without vision when guided by virtual sound versus spatial language*, J. Exp. Psychol.: Applied 12(4), 223‑232. [PDF](https://umaine.edu/vemi/wp-content/uploads/sites/220/2016/06/Klatzky-etal2006-Cognitive-Load-of-Navigating-Without-Vision-jep-a.pdf)
  - Participants follow a virtual path. Guidance is either by words ("left / right / straight ahead") or by a virtual sound placed in the direction to follow. A vibrotactile N‑back task adds mental load.
  - **Without load: no difference. With load: with the virtual sound, the route is shorter and faster, and the secondary task is better performed.**
  - Explanation: a spatialised sound is processed **directly by perception**. Language must first be decoded, then converted into a direction.
- ✅ **Giudice et al. (2008)**, *Environmental learning without vision: effects of cognitive load on interface design*. [PDF](https://umaine.edu/vemi/wp-content/uploads/sites/220/2016/08/Giudice-etal2008-Environmental-Learning-without-Vision-Effects-of-Cognitive-Load-on-Interface-Design-vision08.pdf)
  - With spatialised sound guidance, the secondary task **does not degrade the mental map** of the place. With voice guidance, errors in returning to the starting point increase clearly. Spatialised sound is also faster and more accurate.
- 📚 **Loomis, Golledge & Klatzky (1998)**, *Navigation system for the blind: auditory display modes and guidance*, Presence 7(2). The founding study of the *Personal Guidance System* (UCSB). The "virtual sound" mode is faster and more appreciated than the voice modes.
- 📚 **Marston, Loomis, Klatzky, Golledge (2006)**, *Evaluation of spatial displays for navigation without sight*, ACM TAP. Spatial displays (virtual sound, haptic pointer) beat voice instructions.
- 📚 **Theoretical framework.** According to **Wickens**' multiple resource theory, two tasks interfere more when they use the same channel. According to **Baddeley**, working memory has a "phonological loop" for verbal content and a "visuo-spatial sketchpad" for space. **Voice therefore competes with conversation, thinking and listening to announcements**, whereas a spatial sound goes through the spatial channel.

### 1.2 In the field, it is more nuanced

- ✅ **Mascetti, Picinali, Gerino, Ahmetovic, Bernareggi (2016)**, *Sonification of guidance data during road crossing for people with visual impairments or blindness*, IJHCS. [arXiv](https://arxiv.org/pdf/1506.07272)
  - The study compares two sonified modes and one voice mode for aligning with a pedestrian crossing.
  - **No mode is best for everyone.** More than two thirds of participants prefer a sonified mode. However, they often ask for **voice cues as backup** and find sounds **harder to decode** than voice.
  - ➜ To remember: **sound for direction, voice on demand for meaning.**
- ✅ **Ahmetovic, Avanzini, Baratè, Bernareggi, Ludovico, Mascetti et al.**
  - *Sonification of rotation instructions to support navigation of people with visual impairment*, PerCom 2019 ([dblp](https://dblp1.uni-trier.de/rec/conf/percom/AhmetovicABBGLM19.html)).
  - Then the long version IJHCS 2023, with 9 visually impaired participants in real conditions ([PDF](https://ludovico.lim.di.unimi.it/download/papers/IJHCS2023.pdf)).
  - This work is about **how to sonify a rotation** (turn X° on the spot) and walking in a straight line. It is very close to VisionAid's "orient towards the object" problem. **To read in full, it is a priority.**
- ✅ **Ahmetovic, Guerreiro, Ohn-Bar, Kitani, Asakawa (2019)**, *Impact of Expertise on Interaction Preferences for Navigation Assistance of Visually Impaired Individuals*, W4A. [DOI 10.1145/3315002.3317561](https://doi.org/10.1145/3315002.3317561)
  - The need for detail changes with the person and with whether they already know the route. The authors recommend **adjustable verbosity** to reduce distractions.
  - Twin study at IUI 2018, *Modeling Expertise in Assistive Navigation Interfaces for Blind People*: users improve as they repeat the same route.
- 📚 **NavCog (CMU, Asakawa)**: indoor navigation with BLE beacons, turn-by-turn voice instructions. The reference for all-voice guidance.
- 📚 **Gaunet & Briffault (2005)**: how to phrase verbal instructions for blind pedestrians. "Clock face" references ("at 2 o'clock") are widely used in orientation and mobility (O&M instructors).

### 1.3 Sound or vibration in noise

- ✅ **Bharadwaj, Shaw & Goldreich (2019)**, *Comparing Tactile to Auditory Guidance for Blind Individuals*, Frontiers in Human Neuroscience. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6930908)
  - 12 blind participants compare a vibrating belt and auditory instructions, with and without simulated street noise.
  - In quiet, audio is better. **With street noise, the gap narrows clearly**: auditory performance drops more.
  - Earlier results: Gustafson-Pearce et al. (2007) find fewer errors with touch in noise; Flores et al. (2015) find touch slower.
  - ➜ **A haptic channel in addition to sound** (phone vibration) becomes interesting in noisy environments.

### 1.4 Menu sounds (earcons, auditory icons, spearcons)

- 📚 **Earcons** (Blattner 1989, Brewster): abstract musical patterns. They are expressive but **long and frustrating to learn**.
- 📚 **Auditory icons** (Gaver 1986): everyday sounds (door, footsteps, water). They are **understood without learning**.
- ✅ **Spearcons** (Walker, Nance & Lindsay 2006, ICAD): a synthesised word then **sped up** until it is no longer recognised as speech. For navigating menus, spearcons and voice beat earcons and auditory icons. Palladino & Walker (2007) show that earcons are much harder to learn. [core.ac.uk](https://core.ac.uk/download/4718939.pdf)
- ➜ Spearcons are a **compromise between voice and sound**. Example: "chair" spoken in 150 ms and spatialised.

### 1.5 Comparative summary

| Criterion | Voice | Spatialised sound | Parametric sonification (pitch, tempo…) | Vibration |
|---|---|---|---|---|
| Learning | immediate | fast | longer | medium |
| Cognitive load under dual task | **high** | **low** | low to medium | low |
| Angular accuracy | low ("a bit to the left") | good with head tracking | **very good** if a reference sound indicates the target | medium |
| Information about what is there (meaning) | **excellent** | none | none | none |
| Interpretation latency | 0.5 to 2 s | ~100 ms | ~100 ms | ~100 ms |
| Masks the environment | yes (continuous flow) | little, if sounds are short | depends on timbre | **no** |
| Robustness in noise | medium | medium | medium | **good** |

---

## 2. Which sounds to choose: frequencies, timbre, rhythm

### 2.1 Psychoacoustics of localisation (basics)

- 📚 **ITD (interaural time difference)**: the dominant cue **below ~1.5 kHz**.
- 📚 **ILD (interaural level difference, head shadow effect)**: the dominant cue **above ~1.5 to 3 kHz**.
- 📚 **Spectral cues of the outer ear (pinna)** (for elevation and front/back): they require energy **between 4 and 16 kHz**.
- 📚 Consequence: **a pure sine wave is badly localised**, especially around 2 to 4 kHz, where neither ITD nor ILD are effective (Stevens & Newman 1936). A **broadband, transient sound** (click, filtered noise, sharp attack, percussive sound) provides all the cues at once.
- 📚 **Smallest perceived angle difference**: about 1 to 2° in front, more than 10° on the sides. Small lateral differences are therefore poorly perceived, which justifies the "small-angle exaggeration" already done in VisionAid ([director.py](../../src/visionaid/audio/director.py)).
- ✅ **Walker & Lindsay (2006)**, *Navigation performance with a virtual auditory display: effects of beacon sound, capture radius, and practice*, Human Factors 48. [core.ac.uk](https://core.ac.uk/download/19562103.pdf)
  - **Sonar-type beaconing and a 1.5 m capture radius form the best combination.**
  - Broadband pink noise and the "sonar ping" beat a pure tone.
  - Training clearly improves performance.

### 2.2 Not covering ambient noise

For a blind user, this is **criterion no. 1**. Traffic is used to align, to detect a vehicle pulling away and to know when to cross.

- ✅ Blind pedestrians detect vehicles well in quiet, but badly as soon as there is moderate background noise. Electric vehicles make the problem worse. See [PMC2814321](https://pmc.ncbi.nlm.nih.gov/articles/PMC2814321) and [PMC3046409](https://pmc.ncbi.nlm.nih.gov/articles/PMC3046409).
- ✅ Screen reader users often wear **a single earphone**. Few use bone conduction or transparency mode (Ahmetovic et al. 2023, [PDF](https://iris.unito.it/bitstream/2318/2016332/1/ahmetovic2023enhancing.pdf)).
  - ⚠️ Problem for binaural spatialisation: with a single earphone, it no longer works. **A mono fallback mode is needed**, where direction is carried by pitch or rhythm, not by stereo.

**Leads from psychoacoustics (📚, to validate by tests):**

| Principle | Justification |
|---|---|
| **Short and sparse sounds** (50 to 200 ms, silence in between) rather than continuous | Masking is temporal: a brief sound lets listening to the environment "breathe". |
| **Put the energy between ~1 and 4 kHz, without strong bass** | Urban noise is concentrated below 1 kHz (engines, rolling). A cue between 1 and 4 kHz stands out without covering the low frequencies that carry vehicle information. The ear is also most sensitive around 2 to 4 kHz. |
| **Keep harmonics or high-frequency noise** (above 4 kHz) for localisation | See 2.1: pinna cues are needed to distinguish front from back. |
| **Avoid the speech band (300 Hz to 3 kHz) for background sounds** | So as not to disturb a conversation or the voice announcements. A compromise to find with the previous line. |
| **Volume adapted to ambient noise** (measured by the phone microphone) | Aim a few dB above the masking threshold, no more. |
| **Lower the guidance when the voice speaks (ducking)** | Clear hierarchy: alert > voice > guidance > ambience. |
| **"Natural", non-aggressive timbres** for guidance; **dissonant or rough timbres** reserved for alerts | Research on perceived urgency (below) and on fatigue. |

### 2.3 Urgency, annoyance and natural mappings

- ✅ **Edworthy, Loxley & Dennis (1991)**, **Hellier, Edworthy & Dennis (1993)** and **Hellier & Edworthy (1999)**, on the urgency of alarms. [Plymouth](https://researchportal.plymouth.ac.uk/en/publications/designing-urgency-into-auditory-warnings-using-pitch-speed-and-lo/)
  - Perceived urgency increases with **repetition rate (the most effective lever)**, pitch and inharmonicity.
  - Some parameters increase urgency **without increasing annoyance too much**. That is exactly what to aim for in guidance that lasts for hours.
- ✅ **Suied, Susini & McAdams (2008)**, JEP: Applied: reaction time depends on the urgency of the alarms. [PDF](https://www.mcgill.ca/mpcl/files/mpcl/suied_2008_jepappl.pdf)
- 📚 **Mappings understood without learning:**
  - high pitch = high up (Pratt effect, useful for elevation: step, obstacle at head height);
  - fast = close or urgent (parking sensor);
  - loud = close;
  - consonant = good heading, dissonant = deviation (used by Project Guideline).

### 2.4 Sonification strategies for guidance

- ✅ **Parseihian, Gondre, Aramaki, Ystad, Kronland-Martinet (2016)**, *Comparison and evaluation of sonification strategies for guidance tasks*, IEEE Trans. Multimedia (DOI 10.1109/TMM.2016.2531978). [PDF](https://typeset.io/pdf/comparison-and-evaluation-of-sonification-strategies-for-1dpfi7jxvq.pdf)
  - Strategies are classified into three families: **basic** (varying pitch, volume, tempo…), **with reference** (a sound represents the target and the user makes the current sound converge towards it) and **metaphorical**.
  - **Strategies with reference are more accurate but slower.**
  - To perceive distance, **a non-spatial sonification does better than a spatial one**.
  - ➜ **Spatialisation is for direction, a parameter (tempo, pitch) is for distance.** That is already what VisionAid does with note spacing. This paper confirms it.
- ✅ **Ziemer & Schultheis (2019)**, *A psychoacoustic auditory display for navigation*, J. Multimodal User Interfaces ([arXiv 1912.08609](https://arxiv.org/pdf/1912.08609.pdf)), then the 3D extension ([arXiv 1912.00766](https://web3.arxiv.org/pdf/1912.00766)).
  - Two dimensions of space are coded by **perceptually independent sound attributes** (for example beating or roughness for one axis, brightness or pitch for the other), **in mono**.
  - The route is as short as with vision, but slower. The approach comes from guided surgery.
  - ➜ **Useful for the mono mode with a single earphone.**
- 📚 **The "Geiger" counter or parking sensor** (beep rate proportional to proximity) is the most intuitive scheme for distance. Apple uses it in "People Detection" with the LiDAR (faster beeps when getting closer).

---

## 3. Sound in space: binaural, HRTF, ambisonics… and "8D"

### 3.1 The techniques

| Technique | Principle | For headset guidance |
|---|---|---|
| **Stereo panning** (left/right by volume) | ILD only | Simple, but the sound stays "inside the head" and front cannot be told from back. |
| **"8D audio"** | Rotating panning + reverb + sometimes a bit of HRTF, on music | **Marketing effect, no study evaluates it.** No value for guidance. |
| **Parametric binaural** (Woodworth ITD + Brown & Duda head shadow) | Spherical head model | **This is what VisionAid uses** ([binaural.py](../../src/visionaid/audio/binaural.py)). Light and deterministic. Weak for front/back and elevation. |
| **Generic HRTF binaural** (KEMAR, CIPIC, SOFA database) | Convolution with measured head responses | Adds pinna cues, hence improves front/back and elevation. Tools: Steam Audio, Resonance Audio, 3D Tune-In Toolkit, Web Audio `PannerNode` in `HRTF` mode, Apple PHASE / AVAudioEnvironmentNode. |
| **Personalised HRTF** | Measurement or estimation from photos of the ear (Apple's "Personalized Spatial Audio" with the TrueDepth camera) | Modest gain when the head is tracked (see 3.2). |
| **Ambisonics** (order 1 to 3), then binaural decoding | Representation of the whole sound field | Practical for a **full scene** (several objects) and for head rotation. |
| **Dynamic rendering with head tracking** | The sound stays fixed in the world when the head turns | **The most important factor**, see below. |

### 3.2 What really improves localisation

- ✅ **Head tracking** is "fundamental" for reducing front/back confusions and the "inside the head" sensation. It matters more than HRTF personalisation (RWTH Aachen 2020; arXiv 2510.09161, 2025: with head movements, almost no front/back confusion left, whatever the rendering). [arXiv](https://arxiv.org/pdf/2510.09161)
- ✅ A **well-chosen generic HRTF** does almost as well as a measured HRTF. The KEMAR head is rather poor: 17% front/back confusions, against about 8 to 9% (arXiv 2606.30114, 2026). [arXiv](https://arxiv.org/pdf/2606.30114)
- ✅ **Correcting the spectrum of a non-personalised HRTF** reduces the error by 22% and front/back confusions by up to 66% (Ngan et al. 2010; So et al. 2011).
- 📚 **Short training** durably improves localisation with an HRTF that is not one's own (Zahorik et al. 2006, effect kept for 4 months).
- ➜ **For VisionAid:** the phone is worn on the chest or in the hand, so the IMU gives the body heading, not the head heading.
  - Earphones with a motion sensor (AirPods via `CMHeadphoneMotionManager`, some Android ones) would allow **real head tracking**.
  - Otherwise, encourage the user to **"sweep by turning the body"** to remove the front/back ambiguity.

### 3.3 Bone conduction and open earphones

- ✅ Localisation with bone conduction (transducer in front of the ear) is **comparable to that of a classic headset** (Human Factors 2015). It is worse if the transducer is placed behind the ear. An "attraction" effect towards the direction of the transducer is also observed (Applied Ergonomics 2017, [PubMed 28237013](https://pubmed.ncbi.nlm.nih.gov/28237013/)).
- ✅ With SWAN in bone conduction, all participants succeed in the tasks, with a short adaptation time ([Georgia Tech](https://repository.gatech.edu/entities/publication/d22df9f0-06df-4188-8e2b-f1efcc1567fc)).
- ✅ There is a binaural bone-conduction navigation device with obstacle detection (Applied Sciences 2021, [MDPI](https://www.mdpi.com/2076-3417/11/8/3356)).
- ⚠️ The frequency response of bone conduction is **irregular**: little bass, resonances. **VisionAid's sounds must be tested on a real bone-conduction headset** (Shokz…) and not only on a classic headset.

### 3.4 Perceived distance

- 📚 The natural cues are volume, the ratio between direct sound and reverberation, and high-frequency filtering with distance. With headphones, distance is **poorly perceived** (Kolarik et al. 2016, review). See also arXiv 2505.18020 (2025) on reverberation and distance. This confirms Parseihian: **code distance with a parameter** (tempo), not with spatialisation.

---

## 4. Existing systems and products

### 4.1 Guidance and navigation

| System | Type of sound | What to take from it |
|---|---|---|
| ✅ **Microsoft Soundscape** (2018‑2023, discontinued, **open source MIT code**) | Looping 3D audio beacon placed on a destination, spatialised voice announcements of nearby places | The reference for "sound anchored in the world". Taken over by **VoiceVista** (iOS) and *Soundscape Community*. [MS Research](https://www.microsoft.com/en-us/research/?p=605433) |
| ✅ **Google Project Guideline** (2020, **open source** in Nov. 2023) | A sound becomes louder and **more dissonant in the ear on the side you drift towards**; other frequencies announce a turn; "STOP" signal if the line is lost; bone conduction | Very close to VisionAid's logic. The **code and the 3D simulator** are public. [Google Research](https://research.google/blog/open-sourcing-project-guideline-a-platform-for-computer-vision-accessibility-technology/) |
| ✅ **SWAN** (Georgia Tech, Walker) | Non-verbal spatialised beacons, bone conduction | Research basis on choosing beacon sounds and capture radius. [SWAN 2.0](https://wirelessrerc.gatech.edu/sites/default/files/publications/walker_swan2-wirelessrerc-sot2021-proceedings-v16-withalttext.pdf) |
| ✅ **biped.ai** (Switzerland, CES 2022) | Harness with 3D cameras over 170°, **obstacle trajectory prediction**, spatial audio alerts, bone conduction | Only signals **what crosses the user's path** (a bike at 12 m on the road, but not a nearby object with no risk). A strong idea. [designboom](https://www.designboom.com/technology/biped-ai-copilot-blind-visually-impaired-people-ces-2022-01-10-2022/) |
| 📚 **BlindSquare, Lazarillo, Ariadne GPS** | Voice, plus a few sounds | GPS place announcements, widely used. |
| 📚 **Apple Loupe / Magnifier** (people and door detection) | Beeps faster and faster when getting closer, plus voice | The mainstream "parking sensor" model. |
| 📚 **Seeing AI, Be My AI, Envision** | Voice (AI description) | No continuous guidance, description. |
| 📚 **NavCog** (CMU) | Turn-by-turn voice | Reference for all-voice indoors. |

### 4.2 Sensory substitution ("seeing with the ears")

- 📚 **The vOICe** (Meijer 1992): the image is scanned from left to right, the height in the image gives the pitch, brightness gives the volume. Very powerful, but **tens of hours of training** (✅ about 70 h in a published protocol) and a continuous, **tiring** sound.
- ✅ **EyeMusic** (Amedi): same principle, but with musical notes, with **colour coded by instrument**. More pleasant to listen to ([Frontiers 2023](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1058617/pdf)).
- ✅ **Topo-Speech** (2023): the position of an object is given by a spatialised voice saying its name. About 74% success among blind people **after a single session**. ➜ A very promising **voice + space fusion**.
- ✅ **Effect of ambient noise on these devices**: [HAL Sorbonne](https://hal.sorbonne-universite.fr/hal-02124943/file/) (*Multisensory Research*). Performance drops with irrelevant noise.
- 📚 **Sound of Vision** (European H2020 project, 2015‑2018): depth camera, **audio and haptic** rendering of the scene, several sonification modes.
- 📚 **Naviton** ✅ (Bujacz & Strumiłło): sonification of the 3D scene, tested with blind people.
- 📚 **Human echolocation** (Thaler, Daniel Kish): tongue clicks give a rich perception of space. There is work on **assisted** or synthetic **echolocation**.

### 4.3 Reviews to read first

1. ✅ **Bujacz & Strumiłło (2016)**, *Sonification: Review of Auditory Display Solutions in Electronic Travel Aids for the Blind*, Archives of Acoustics 41(3), 401‑414. [PDF](https://www.journals.pan.pl//Content/102689/PDF/aoa-2016-0040.pdf). **The reference review** on our topic.
2. 📚 **Csapó, Wersényi, Nagy, Stockman (2015)**, *A survey of assistive technologies and applications for blind users on mobile platforms*, J. Multimodal User Interfaces.
3. 📚 **Dakopoulos & Bourbakis (2010)**, *Wearable obstacle avoidance electronic travel aids for blind: a survey*, IEEE SMC‑C.
4. 📚 **Hermann, Hunt, Neuhoff (2011)**, *The Sonification Handbook*. **Free online**, chapters on earcons, auditory icons and navigation.
5. ✅ *Spatial Knowledge via Auditory Information for Blind Individuals: Spatial Cognition Studies and the Use of Audio-VR*. [PMC9268803](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9268803/)
6. ✅ *Spatial navigation with horizontally spatialized sounds in early and late blind individuals*. [PMC7909643](https://pmc.ncbi.nlm.nih.gov/articles/PMC7909643/). Differences between early and late blind people matter for the choice of sounds.
7. 📚 **Kolarik et al. (2016)**, *Auditory distance perception in humans: a review*, and their review on echolocation.
8. ✅ 2025 design framework: *Cognitive Load-Aware Framework for Accessible Urban Mobility*. [MDPI](https://www.mdpi.com/2673-2688/7/6/206). Recommends "selective, spatially coded, calibrated" feedback rather than talkative feedback.

### 4.4 Where to find user testimonials

- **AppleVis** (forums and podcasts by blind iOS users, many reviews of Soundscape and VoiceVista).
- **Soundscape Community mailing list and community**.
- **"Double Tap" (AMI Audio) and "Eyes On Success" podcasts**.
- **Associations in France**: Fédération des Aveugles de France, **Valentin Haüy** (its technology department), **Voir Ensemble**, and **orientation and mobility instructors** (ARAMAV, rehabilitation centres). These are the best sources for organising interviews.
- ➜ Recommendation: **10 to 15 short interviews** (early and late blind, visually impaired, guide dog or cane users) before freezing the sound design.

---

## 5. Where VisionAid stands compared with this state of the art

| Current VisionAid choice | What the literature says |
|---|---|
| Spatialised note towards the goal, interval shrinking when getting closer | ✅ Consistent: direction by spatialisation, distance by tempo (Parseihian 2016; parking sensor). |
| Exaggeration of small angles | ✅ Justified by the poor lateral discrimination. Compare with Project Guideline's "deviation exaggeration". |
| Parametric binaural (Woodworth + Brown‑Duda), muffled rear | 🟡 Good basis. A measured HRTF (SOFA) and above all **head tracking** would improve front/back. |
| Distinct, softer exploration sound | ✅ Consistent with the hierarchy and with the goal of reducing annoyance. |
| Low, non-spatialised, repeated alert | 🟡 A **low** sound comes through badly in bone conduction and gets confused with traffic noise. Urgency comes through better with **tempo** and roughness. To test. |
| Arrival: arpeggio or voice "vous êtes arrivé" (you have arrived) | ✅ Hybrid voice + sound, consistent. |
| Single-earphone mode | ❌ Missing. Many users wear a single earphone. |
| Adaptation to ambient noise | ❌ Missing. |

> Note from the fork: after this research, the team switched the phone rendering from HRTF to a constant-power stereo pan, a single guidance note with exaggerated small angles, and a separate soft exploration sound; see [`DECISIONS.md`](DECISIONS.md).

---

## 6. Innovative leads for VisionAid

Ranked from the simplest to the most forward-looking. Each lead can be tested before being coded.

### A. Managing attention

1. **Guidance by exception ("silence = all is well").** When the heading is right, nothing, or a very discreet tick at the pace of the steps. Sound only appears on deviation, obstacle or arrival. Fatigue is reduced and listening is freed.
2. **Adaptive verbosity.** The system starts with voice + sound, then gradually moves to sound only as the user becomes an expert or repeats the same route (Ahmetovic 2018‑2019). An on-demand "What is it?" button triggers the voice (Mascetti 2016).
3. **Strict sound hierarchy with ducking**, on the model of aviation radio: alert > voice > guidance > ambience. One message at a time.
4. **Estimated cognitive load** from walking speed, hesitations and ambient noise. When the load rises, switch to spatialised sound only (Klatzky 2006).

### B. Staying audible without covering the environment

5. **Placing sounds in the "holes" of the ambient spectrum.** The phone microphone analyses the noise (spectrum by third-octave bands). The timbre or filter of the guidance sound is shifted into the least occupied bands, like birds singing in the "acoustic niches" left free (Krause hypothesis). **A lead little explored in the literature on aids for blind people. Very innovative.**
6. **Volume adapted to masking**: a few dB above the computed masking threshold, instead of a fixed volume.
7. **Protecting critical moments.** During a crossing (detected or declared), everything except alerts is cut, to leave listening to traffic intact.
8. **Sounds designed for bone conduction**: energy between 1 and 4 kHz, no significant bass, sharp attacks.

### C. Coding space better

9. **World-anchored sound with head tracking** (AirPods, or phone worn as a pendant): the sound stays fixed on the object when the head turns. The main lever for front/back.
10. **Mono mode without stereo** for the single-earphone user. Psychoacoustic coding "à la Ziemer": a deviation to the left gives a slow beating, a deviation to the right a brightness, the right heading a pure, consonant sound.
11. **Target reference ("strategy with reference")**: a fixed reference sound represents the target and the current sound "tunes" to it while turning. Consonance gives the right heading, like tuning a guitar. More accurate according to Parseihian 2016, and used as dissonance by Project Guideline.
12. **Elevation by pitch** for steps, kerbs and obstacles at head height (high pitch = high up). Cameras often handle only the horizontal plane.
13. **Doppler and tempo for time to collision**: an approaching object has its rhythm speed up according to the **time to contact**, not the distance (biped's approach). Only trajectories that cross the user's are signalled.

### D. Adding meaning without a talkative voice

14. **Spatialised spearcons**: "door", "chair", "stairs" in 150 to 250 ms, placed at the location of the object (in the line of Topo-Speech). Meaning is kept while the duration is divided by 3 or 4.
15. **Auditory icons per object class**: a light creak for a door, a clink for a cup… Understood without learning (Gaver). Can be combined with a timbre per category (on the model of the instrument for colour in EyeMusic).
16. **Description by a language model, but short, on demand and spatialised**: "three chairs, a table at the far right". This complements guidance, without replacing it.

### E. Thinking outside the box

17. **Music synchronised with walking**: the IMU pedometer sets the guidance tempo to the step cadence, and the heading deviation changes the harmony. Less tiring than a series of beeps. There is work in rehabilitation (rhythmic auditory stimulation), to explore for guidance.
18. **Virtual echolocation**: a synthetic click emitted by the user on request; the 3D scene seen by the camera returns synthetic "echoes" (delay and filtering according to distance and material). It builds on a skill some blind people already master (Thaler, Kish).
19. **"Sound torch"**: by pointing the phone, you hear what is in the camera cone (a vOICe-like sonification but **limited to detected objects**, hence not very noisy).
20. **Sound and vibration together**: direction by sound, alert or validation by vibration, for noisy environments (Bharadwaj 2019).
21. **Personalisation through mini-games**: 2 minutes of a localisation game at first launch. The best-perceived HRTF among 5 to 10 is chosen, angle exaggeration and volume are calibrated, and the user is trained (learning effect demonstrated: Zahorik, Walker & Lindsay).

---

## 7. Proposed mini-experiments (before coding)

| # | Question | Light protocol |
|---|---|---|
| E1 | Voice or sound to orient towards an object, with a secondary task (counting backwards) | Take Klatzky 2006's idea: 8 to 10 subjects, measure time, angle error and success of the secondary task. |
| E2 | Which timbre is best localised **with bone conduction in street noise** | 4 timbres (sine, 1‑4 kHz filtered noise, click, note rich in harmonics) × 8 directions. Measure the error and front/back confusions. |
| E3 | How much does it hinder vehicle detection | Recorded vehicle pass-by sounds + guidance on or off. Measure the detection rate. |
| E4 | Mono mode, single earphone | Ziemer coding vs panning. |
| E5 | Acceptability over a long period | 20 min of walking, NASA‑TLX scale and annoyance questionnaire. |

---

## 8. Going further: tiring or non-tiring sounds, sounds for visually impaired people, frequencies

### 8.1 What makes a sound tiring or unpleasant

**Psychoacoustic measures of annoyance (Zwicker & Fastl)**
- ✅ The *psychoacoustic annoyance* model combines four quantities ([Fastl, TUM](https://mediatum.ub.tum.de/download/1138438/1138438.pdf)):
  - **loudness** (perceived volume, ISO 532‑1/2);
  - **sharpness** (a lot of energy in the treble; it only penalises above about 1.75 acum);
  - **roughness** (fast modulation, maximum around 70 Hz);
  - **fluctuation strength** (slow modulation, maximum around 4 Hz).
- ✅ **Tonality** (pure or whistling sound) is missing from Zwicker's model. NASA added it in 2024 because it strongly increases annoyance ([NASA TM‑20240003202](https://human-factors.arc.nasa.gov/publications/NASA-TM-20240003202June.pdf)).
- ➜ These quantities can be **computed** (MoSQITo libraries, Python). Each VisionAid sound can therefore be **scored objectively** before testing it on people.

**"Screeching" sounds**
- 📚 **Kumar, Forster, Bailey & Griffiths (2008)**, *Mapping unpleasantness of sounds to their auditory representation*, JASA.
- 📚 **Kumar et al. (2012)**, *Features versus feelings*, J. Neurosci.
- Result of these two studies: the sounds judged most unpleasant (chalk on a blackboard, screeching) concentrate their energy **between ~2.5 and 5.5 kHz** and have a **temporal modulation between ~1 and 16 Hz**.
- 📚 Halpern, Blake & Hillenbrand (1986): removing the treble from a chalk screech **does not make it less unpleasant**. It is the mid frequencies (2 to 4 kHz) that disturb.
- ⚠️ This zone overlaps the one where the ear is most sensitive and that is recommended for audibility (section 2.2). **Energy is needed around 2 to 4 kHz, but without rough modulation or "screeching" content.** A harmonic sound, with a soft attack and a natural decay, is fine; a modulated grating noise is not.

**Consonance and timbre**
- ✅ Marjieh et al. (2024, *Nature Communications*) show that **timbre changes which intervals are judged pleasant**: consonance is not only a matter of frequency ratios. [PMC11258268](https://pmc.ncbi.nlm.nih.gov/articles/PMC11258268)
- ✅ Sounds with a **harmonic** spectrum (like instruments or the voice) are the basis of the sensation of consonance (Bowling & Purves, [PMC2885564](https://pmc.ncbi.nlm.nih.gov/articles/PMC2885564)).
- ✅ For sine waves, **preference drops as the mean frequency rises**, for all intervals and all groups tested ([arXiv 2510.01201](https://arxiv.org/pdf/2510.01201v1)).
- ➜ For background guidance, **harmonic, rather mid-range notes** are better than pure high beeps.

**Repetition, habituation and "alarm fatigue"**
- ✅ In research on medical alarms, **introducing variation in the rhythm** of the sounds reduces annoyance **without harming their recognition** ([PMC3628049](https://pmc.ncbi.nlm.nih.gov/articles/PMC3628049)).
- ✅ **Instrument or percussive timbres** stay more useful in loud environments and limit alarm fatigue ([Scientific American](https://www.scientificamerican.com/article/making-alarms-more-musical-can-save-lives/); McMaster thesis 2024, [Mattar](https://macsphere.mcmaster.ca/bitstream/11375/29792/1/Mattar_Maya_2024.pdf)).
- ✅ **Any sound ends up being ignored if it goes off wrongly too often.** Fatigue depends on the *false alarm rate* as much as on the sound itself.
- ➜ For VisionAid, the **reliability of obstacle detection** therefore matters as much as the choice of alert sound.

**Monotony and listening time**
- ✅ Blind users describe **mental fatigue with the "flat" voices** of screen readers. More expressive narration (VoxVista, 20 blind participants) improves engagement and understanding ([Bragg CHI 2018 and related work](https://wildlab.cs.washington.edu/Publications_files/Bragg_CHI2018.pdf), [ODU](https://digitalcommons.odu.edu/computerscience_fac_pubs/442)).
- ✅ In security operations centres, **listening fatigue** limits how long a sonification can be tolerated. **Occasional** use seems more realistic than continuous listening ([arXiv 1807.06706](https://arxiv.org/pdf/1807.06706)).
- ✅ The **aesthetics** of an auditory display play a direct role in fatigue and annoyance, but a sound that is too "musical" can distract from the information ([arXiv 1311.5426](https://arxiv.org/pdf/1311.5426)).
- ✅ **vOICe (sensory substitution)**: very few users adopt it over time, only a handful of experts. The difficulties cited are **separating several simultaneous sounds**, overload in cluttered scenes and the length of training ([SoundSight, JMUI 2021](https://link.springer.com/article/10.1007/s12193-021-00376-w); [Bertram & Stafford 2016](https://eprints.whiterose.ac.uk/101090/1/bertram16.pdf)).
- ➜ **One sound at a time, a few sources at most, silences.**

**Classic alarm design rules (📚, to check)**
- **Patterson (1982, Civil Aviation Authority)**:
  - pulses of **100 to 150 ms** with **rise and fall of 20 to 30 ms**, to avoid the startle effect;
  - at least **4 harmonics between 0.5 and 4 kHz**;
  - level **15 to 25 dB above the masking threshold**, no more (beyond that, the sound frightens and annoys).
- **ISO 7731** (danger signals): components between **500 and 2,500 Hz**, with energy below 1,500 Hz for hearing-impaired people. The signal must exceed the masking threshold by about 13 dB per critical band, or 15 dB per third octave.
- **IEC 60601‑1‑8** (medical alarms): fundamental between **150 and 1,000 Hz**, at least 4 harmonics between 300 and 4,000 Hz.

### 8.2 Sounds designed for blind people: accessible pedestrian signals

These are the most standardised and most tested sounds designed for blind people. They are a direct reference for VisionAid.

- ✅ **"Walk" signal** (US MUTCD standard, states): **several frequencies, dominant at 880 Hz**, 0.15 s pulses repeated every 0.15 s (8 to 10 ticks per second). [Delaware](https://regulations.justia.com/states/delaware/title-2/2000/2400/2402/part-4/chapter-4e/section-4e-11)
- ✅ **Pushbutton locator tone**:
  - **0.15 s at most, once per second**;
  - **only 2 to 5 dB above ambient noise**, and **automatically adjusted to the noise**;
  - audible only within **2 to 4 m**;
  - one manufacturer uses a softened 800 Hz square wave or a **50 ms click**.
  - Sources: [walkinginfo.org](https://www.walkinginfo.org/aps/3-7), [Access Board](https://www.access-board.gov/files/research/APS-common-problems.pdf).
- ➜ **This is exactly the philosophy recommended above**: a brief, sparse sound, just above the noise and adapting to it. It can directly inspire VisionAid's "beacon sound".
- 📚 In France, accessible pedestrian signals are triggered on demand by a **universal remote control** (standard NF S 32‑002) and give a voice message or a sound. A good example of **sound on demand** rather than permanent sound.

### 8.3 Frequencies: summary with figures

| Frequency zone | Role / finding | Consequence for design |
|---|---|---|
| **50–200 Hz** | Tyre rolling, heavy trucks, intersections (✅ [JAES 2025](https://reference-global.com/2/v2/download/article/10.2478/jaes-2025-0037.pdf)). Poorly reproduced by bone conduction. | Put no information there. |
| **300–1,000 Hz** | **Peak of mixed traffic noise** (✅ [Environment International 2019](https://doaj.org/article/50e5b3b2247e4595966c53766caf5bc0): average maximum at 1 kHz). Effective ITD cue. Pedestrian signals at 880 Hz. | A fundamental in this zone is audible and localisable, but **competes with traffic**. Harmonics above are needed. |
| **1–2 kHz** | ITD/ILD transition zone. Speech band. | Good zone for the **body** of a guidance sound, to lower while the voice speaks. |
| **2–4 kHz** | **Maximum sensitivity of the ear**. Most "unpleasant" zone if the sound is modulated or rough (📚 Kumar). | Put **harmonics** here for audibility, **without roughness**. Keep roughness for alerts. |
| **4–16 kHz** | **Pinna cues**: front/back, elevation. **First zone lost with age** (presbycusis). Limited in bone conduction. | Useful for localisation in young people. **Do not rely on it alone**. |

- ✅ **Bandwidth**: stimuli **wider than one octave** are localised best (error of about 6 to 7°), whatever their centre frequency. Pure tones give **lower accuracy and large differences from one person to another** ([Yost & Zhong, bandwidth](https://www.bohrium.com/paper-details/sound-source-localization-identification-accuracy-bandwidth-dependencies/814683735180967936-582)).
- ✅ A **100 Hz click train** becomes **less well localised above about 70 dB**, whereas broadband noise stays stable ([Georgia Tech](https://repository.gatech.edu/entities/publication/51cdb3b4-ceff-4839-b34a-1b34452643be)). For a rapidly repeated click, it is mostly the **first click** that gives the direction (precedence effect, [PMC3254714](https://pmc.ncbi.nlm.nih.gov/articles/PMC3254714)).

### 8.4 An often forgotten point: many visually impaired people also hear less well

- ✅ Among people over 71 in the United States (NHATS study, with objective tests), **22% have a dual sensory loss** (vision and hearing), and **59% after 90**. [PMC8601132](https://pmc.ncbi.nlm.nih.gov/articles/8601132)
- ✅ The most frequent combination of sensory impairments is **hearing + vision** (17% in the BLSA cohort, 50% in ARIC). [PMC10922129](https://pmc.ncbi.nlm.nih.gov/articles/PMC10922129/)
- 📚 Visual impairment mainly affects older people (AMD, glaucoma). **Presbycusis** first causes the loss of **high frequencies (above 2 to 4 kHz)** and is often accompanied by **recruitment**: loud sounds quickly become painful.
- ➜ Consequences:
  1. do not code important information **only in the treble**;
  2. avoid **loud volumes** and prefer a volume set according to ambient noise;
  3. allow **adjustment of the frequency curve** (or rely on the phone's audio accessibility settings);
  4. take **hearing aids** into account (Bluetooth streaming to the hearing aid; spatialisation may then be distorted).

### 8.5 Concrete recommendations for VisionAid's sounds

| Parameter | Recommendation | Basis |
|---|---|---|
| Duration of a guidance sound | 80 to 200 ms | Patterson, pedestrian signals |
| Envelope | 10 to 30 ms attack, natural decay (percussive or plucked), never a sharp edge | Patterson; avoids clicks and startle |
| Spectrum | Fundamental 400 to 1,000 Hz + harmonics up to 4 to 8 kHz (bandwidth over one octave) | Broadband localisation; robust to presbycusis and bone conduction |
| Roughness / modulation | None for guidance; **reserved for alerts** | Zwicker/Fastl; Kumar; urgency (Edworthy) |
| Level | 3 to 10 dB above the measured ambient noise, capped | Pedestrian signals (2 to 5 dB); Patterson (at most 15 to 25 dB for alerts) |
| Repetition | No continuous sound. Rhythm linked to distance. **Silence when the heading is right** | Listening fatigue; habituation |
| Variation | Slight variations of timbre, pitch or rhythm from one sound to the next | Medical alarms: less annoyance, recognition intact |
| Pitch | Mid-range rather than high for background | Preference decreasing with frequency (sine waves) |
| Alert | Faster, rougher or dissonant, but brief. Low pitch alone should be avoided (masked by traffic, poorly reproduced by bone conduction) | Edworthy; JAES 2025 |
| Evaluation | Automatic computation of loudness, sharpness, roughness and tonality (MoSQITo) + NASA‑TLX + annoyance questionnaire after 20 min | Zwicker/Fastl; NASA 2024 |

---

## 9. What existing products do with voice and sound (November 2026)

> Sources: manufacturers' websites, tech press, user forums (AppleVis). Forums give **testimonials**, not measurements. Features change quickly; check again before citing.

### 9.1 Smart glasses: voice description, not guidance

| Product | What it does on the audio side | Guidance? |
|---|---|---|
| ✅ **Envision Glasses + "Ally" assistant** | **Conversational voice assistant**: you ask it a question out loud and can interrupt it. Text reading, face recognition, scene description, quick answers. Ally has been available on all editions since August 2025. | **No.** Contextual help (airport menus, seat controls…), no step-by-step guidance. Envision **advises against using glasses as the main navigation tool**. [Envision](https://www.letsenvision.com/blog/ally-ai-assistant-sightcity-2025), [support](https://support.letsenvision.com/hc/en-us/articles/32029669353873-Ally-on-Envision-Glasses), [press 2026](https://tribune.com.pk/story/2585783/smart-glasses-find-purpose-among-blind-users) |
| ✅ **Ally Solos** (Envision + Solos) | Same Ally assistant on lighter audio glasses. | **No built-in navigation yet** according to the team (AppleVis Extra #110). One comparison mentions instructions relayed from the phone's navigation apps. [AppleVis](https://www.applevis.com/podcasts/applevis-extra-110-envision-ally-solos-smart-glasses), [Android Police](https://www.androidpolice.com/ally-solos-glasses-use-ai-to-help-blind-people-navigate-the-world/) |
| ✅ **Ray-Ban / Oakley Meta** | Voice commands "look and describe", "look and read". **Call to a Be My Eyes volunteer**, who sees through the camera and answers through the **open speakers**. 📚 Since 2025, a "detailed responses" setting and the "Live AI" mode (continuous conversation) exist. | **No.** The American Foundation for the Blind notes that these glasses **are not designed for blind people** and that Be My Eyes AI did not work on them in autumn 2025. [Be My Eyes](https://www.bemyeyes.com/be-my-eyes-smartglasses/), [AFB](https://afb.org/aw/fall2025/meta-glasses-review), [Discover Sight](https://discoversight.com/meta-ray-ban-glasses-for-blind-users/) |
| ✅ **Google / Android XR (Gemini), pre-orders on 12/11/2026** | **"Guided Vision"**: conversational audio descriptions (small print, finding your keys, room layout, device screens). Access to a **human Aira interpreter** by voice ("Hey Gemini, open Aira Explorer"). **Turn-by-turn Google Maps** directions by voice. | **Explicitly not** for mobility. Google states that Guided Vision is **neither a mobility aid nor a substitute for the cane**, and is used **neither for navigation nor for obstacle detection**. [Google blog](https://blog.google/innovation-and-ai/technology/xr-ar/google-intelligent-eyewear-collections-pre-order/), [Google Store](https://store.google.com/us/magazine/intelligent-eyewear-xr-glasses?hl=en-US), [AppleInsider](https://appleinsider.com/articles/26/11/12/googles-ai-smart-glasses-work-with-iphone-preorders-are-live) |
| ✅ **EchoVision (AGIGA)** | 110° wide-angle camera, **proactive AI** that describes without waiting to be asked, public transport directions, reading, Be My Eyes and Aira integration. **Vibration feedback** for the battery, **sound alert** to find the case. | Mostly transport directions. Users complain about **battery life, price and delays**. [AppleVis](https://www.applevis.com/comment/193969), [Double Tap](https://doubletaponair.com/weekend-exploring-the-features-of-agiga-echo-vision-wearable-tech-2/), [NextBigFuture](https://www.nextbigfuture.com/2026/04/echovision-ai-glasses-for-the-blind-and-low-vision-users.html) |
| ✅ **Aira** (service) | A **human agent** sees the video stream of the glasses or phone and guides **by voice**. | Yes, but by a human, paid and verbal. |
| 📚 **OrCam MyEye** | Clip on the glasses arm: reading by pointing a finger, faces, banknotes, objects. **Voice only.** | No (an orientation feature was announced, status uncertain). |

**To remember about glasses.** All the major glasses (Meta, Google, Envision, Solos, EchoVision) are **centred on voice and on-demand description**, by AI or by a human. **None offers spatialised sound guidance or a continuous signal towards a target.** Their manufacturers even refuse to present them as mobility aids, for reasons of liability and reliability.

### 9.2 Mobility devices: non-verbal sound, vibration

| Product | Sound / voice | What to take from it |
|---|---|---|
| ✅ **biped NOA** (vest worn like a backpack, cameras over 170°, from head to floor, day and night) | Obstacles in **spatial audio through bone conduction**, coded in 3 dimensions: **volume rises when getting closer, stereo gives the direction, pitch gives the height of the obstacle**. GPS navigation and AI descriptions by voice. Only signals objects **on the trajectory** (motion prediction). | **The product closest to VisionAid.** Mixed user feedback on its **appearance** and on wearing it on the shoulders. One user says the device "vibrated constantly" during their test. [AppleVis SightCity 2025](https://applevis.com/comment/190102), [AppleVis 2024](https://www.applevis.com/comment/172442) |
| ✅ **.lumen Glasses** (Romania, CES 2025-2026) | **Vibrations on the forehead** for direction, even for small changes. **Voice alerts** for situations ("narrow passage"). Speakers aimed at the ears. | Deliberate choice: **touch for continuous guidance, voice for events**. Discreet for people around. High price. [Engadget](https://www.engadget.com/wearables/dotlumens-haptic-headset-could-help-blind-people-navigate-181532099.html), [New Atlas](https://newatlas.com/wearables/dotlumen-ai-glasses-blind-independence), [designboom](https://www.designboom.com/technology/forehead-glasses-assist-blind-people-move-around-without-guide-dog-lumen-12-30-2025/) |
| ✅ **Glidance Glide** (small wheeled robot held by a handle) | Guidance **by vibrations in the handle**, plus a speaker and Bluetooth for audio instructions. A "directed" mode takes the user to a destination. | The robot **physically pulls** the user, so almost no sound is needed for direction. Still in pre-order or beta. [Robot Report](https://www.therobotreport.com/rbr50-company-2024/glidance-provides-independence-to-visually-impaired-individuals/), [Podfeet CSUN 2025](https://www.podfeet.com/blog/2025/06/csun-2025-glidance/), [MS Research](https://www.microsoft.com/en-us/research/?p=911676) |
| ✅ **WeWALK Smart Cane 2** | Overhead obstacles: **beeps that speed up like a parking sensor**, or vibration, as preferred. **AI voice assistant** (ChatGPT): step-by-step route and description of the surroundings. Speaker in the cane, or earphones / **hearing aids** via Bluetooth. | Classic hybrid: **sounds for urgency, voice for meaning**. [Engadget](https://www.engadget.com/home/the-wewalk-smart-cane-2-could-be-one-of-ais-few-good-use-cases-at-ces-2025-182020074.html), [AppleVis](https://www.applevis.com/comment/188319), [RNIB](https://shop.rnib.org.uk/products/wewalk-smart-cane-2-ai) |
| 📚 **Sunu Band** | Ultrasonic wristband, **vibration only**. | Touch without sound. |

### 9.3 Phone apps

| App | Sound / voice |
|---|---|
| ✅ **Seeing AI, "World" channel** (Microsoft, iPhone with LiDAR) | **Audio augmented reality**: objects around you are **announced by a spatialised voice** at their position. "Find My Things" lets you teach it your own objects and then **follow sound cues** to find them. [App Store](https://apps.apple.com/app/id999062298) |
| 📚 **Apple Magnifier / Loupe** (people and door detection; LiDAR) | As preferred: **sounds that speed up when getting closer**, vibrations or spoken distance. Door detection **describes by voice** the door, the handle and the sign. |
| ✅ **Microsoft Soundscape → VoiceVista** | **3D audio beacon** towards a destination and spatialised place announcements. See section 4.1. |
| 📚 **Google Lookout** | Continuous description **by voice** (text, objects, banknotes). No documented spatialisation. |
| ✅ **Project Guideline** (Google) | **Dissonant sound in the ear on the side you drift towards**. See section 4.1. |
| 📚 **BlindSquare, Lazarillo** | Voice (GPS announcements), plus notification sounds. |

### 9.4 What can be concluded for VisionAid

1. **A fairly clear split:**
   - **Glasses = voice and description**, on demand (generative AI or human).
   - **Mobility = non-verbal sound or vibration**, continuously: NOA, .lumen, Glide, WeWALK, Apple's Magnifier.
   - This is exactly what research says (sections 1 and 2): **sound or touch for "where", voice for "what"**.
2. **The little-occupied niche: guiding to a specific object with spatialised sound.**
   - Glasses **say** where the object is ("the keys are on the left on the table") but **do not guide the hand or the step**.
   - Only Seeing AI (Find My Things, with LiDAR) and NOA (for obstacles, not targets) come close.
   - **VisionAid, with continuous spatialised guidance towards a target and cheap hardware, sits in this gap.**
3. **NOA's coding confirms the choices of the literature**: stereo for direction, volume or tempo for distance, pitch for height. A field validation of lead 12 (elevation by pitch).
4. **.lumen's choice** (touch for continuous guidance, voice for events) is a serious alternative to all-sound. It has the advantage of **masking nothing** for the ear (section 1.3).
5. **User complaints** rarely concern the sound itself. They concern **battery life, price, weight or wearability, the way others look at you, and reliability**. A "cheap hardware + phone" solution directly answers several of these points.
6. **Liability**: Google and Envision **refuse** to present their glasses as mobility aids. VisionAid must position itself clearly (a complement to the cane or the dog, not a replacement) and take care of the **reliability of alerts** (section 8.1, alarm fatigue).
7. **A concrete lead**: glasses with **open speakers** (Meta, Google, Solos) have two speakers near the ears and a front camera that follows the head. **They could serve as hardware for VisionAid-style spatialised guidance**, which none of them does today. The Android XR and Meta development kits are worth watching.

---

## Sources consulted (✅)

- Klatzky et al. 2006: https://umaine.edu/vemi/wp-content/uploads/sites/220/2016/06/Klatzky-etal2006-Cognitive-Load-of-Navigating-Without-Vision-jep-a.pdf
- Giudice et al. 2008: https://umaine.edu/vemi/wp-content/uploads/sites/220/2016/08/Giudice-etal2008-Environmental-Learning-without-Vision-Effects-of-Cognitive-Load-on-Interface-Design-vision08.pdf
- Mascetti et al., sonified crossing: https://arxiv.org/pdf/1506.07272
- Ahmetovic et al., rotations IJHCS 2023: https://ludovico.lim.di.unimi.it/download/papers/IJHCS2023.pdf
- Ahmetovic et al., expertise W4A 2019: https://ri.cmu.edu/?p=121341
- Ahmetovic et al. 2023, screen readers and sound environment: https://iris.unito.it/bitstream/2318/2016332/1/ahmetovic2023enhancing.pdf
- Parseihian et al. 2016: https://typeset.io/pdf/comparison-and-evaluation-of-sonification-strategies-for-1dpfi7jxvq.pdf
- Walker & Lindsay 2006: https://core.ac.uk/download/19562103.pdf
- SWAN 2.0: https://wirelessrerc.gatech.edu/sites/default/files/publications/walker_swan2-wirelessrerc-sot2021-proceedings-v16-withalttext.pdf
- Spearcons: https://core.ac.uk/download/4718939.pdf
- Bujacz & Strumiłło 2016: https://www.journals.pan.pl//Content/102689/PDF/aoa-2016-0040.pdf
- Ziemer & Schultheis: https://arxiv.org/pdf/1912.08609.pdf, https://web3.arxiv.org/pdf/1912.00766
- Bharadwaj et al. 2019: https://pmc.ncbi.nlm.nih.gov/articles/PMC6930908
- Bone conduction: https://repository.gatech.edu/entities/publication/d22df9f0-06df-4188-8e2b-f1efcc1567fc, https://pubmed.ncbi.nlm.nih.gov/28237013/, https://www.mdpi.com/2076-3417/11/8/3356
- HRTF and head tracking: https://arxiv.org/pdf/2510.09161, https://arxiv.org/pdf/2606.30114, https://www.ieda.ust.hk/dfaculty/so/pdf/Ngan-et-al-2010-Ergonomics.pdf
- Alarm urgency: https://researchportal.plymouth.ac.uk/en/publications/designing-urgency-into-auditory-warnings-using-pitch-speed-and-lo/, https://www.mcgill.ca/mpcl/files/mpcl/suied_2008_jepappl.pdf
- Project Guideline: https://research.google/blog/open-sourcing-project-guideline-a-platform-for-computer-vision-accessibility-technology/
- Soundscape: https://www.microsoft.com/en-us/research/?p=605433, VoiceVista: https://apps.apple.com/app/6450388413
- biped: https://www.designboom.com/technology/biped-ai-copilot-blind-visually-impaired-people-ces-2022-01-10-2022/
- EyeMusic 2023: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1058617/pdf
- Noise and sensory substitution: https://hal.sorbonne-universite.fr/hal-02124943/
- Blind pedestrians and traffic noise: https://pmc.ncbi.nlm.nih.gov/articles/PMC2814321, https://pmc.ncbi.nlm.nih.gov/articles/PMC3046409
- Spatial cognition and audio‑VR: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9268803/
- Early and late blind: https://pmc.ncbi.nlm.nih.gov/articles/PMC7909643/
- 2025 design framework: https://www.mdpi.com/2673-2688/7/6/206
- Distance and reverberation: https://arxiv.org/pdf/2505.18020
