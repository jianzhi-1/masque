# Masque 🎭

[Application Repo](https://github.com/jianzhi-1/masque-prod)
[Report](https://github.com/jianzhi-1/masque/blob/main/report.pdf)
[Presentation](https://github.com/jianzhi-1/masque/blob/main/presentation.pdf)

Masque is a monotonous-to-expressive audio converter. Pass it a monotonous audio (such as one generated from a Text-To-Speech system like Tacotron2) and an emotion label (say "happy"), and it will produce an audio that expresses that emotion.

On the technical side, Masque is a lightweight transformer encoder model that uses articulatory encodings as intermediate features. The training of Masque is essentially a supervised learning problem. To get around the alignment issue of monotonous and expressive audio pairs, the dynamic time warping algorithm is used on the Mel spectrograms to ensure the resultant spectrograms are of the same shape. The Hi-Fi GAN is used at the end of the pipeline to synthesise the final waveform.

The novel part of this project is the usage of articulatory features as intermediate features. In particular, the naive use of Mel spectrograms themselves are not sufficient to produce high-quality synthesised speech. Also, the entire pipeline does not require any transcript - the problem is solved within the image and articulatory space and does not require any textual elements or annotations.

This repository documents the research infrastructure (models, training, utility functions). See the [app repo](https://github.com/jianzhi-1/masque-prod) for a quickstart on using the model.

_(This project is completed during UC Berkeley Fall 2024's iteration of ELENG 225D Audio Signal Processing.)_

### Set up
```shell
sh setup.sh
```

### Progress
- [x] Phase I: Transformer encoder architecture
- [x] Phase II: GAN-loss
- [x] Phase III: Add CNN preprocessing layers
- [x] Phase IV: SPARC
- [x] Report

### Citations
Please see [report](https://github.com/jianzhi-1/masque/blob/main/report.pdf).