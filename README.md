  Single-object tracking is a fundamental enabling
technology in the field of remote sensing observation. It plays
a crucial role in tasks such as unmanned aerial vehicle route
surveillance and maritime vessel trajectory prediction. However,
because of challenges such as the weak discriminative power
of target features, interference from complex environments, and
frequent viewpoint changes, existing trackers often suffer from
insufficient temporal modeling capabilities and low computa
tional efficiency, which limit their practical deployment. To
address these challenges, we propose TSTrack, a novel lightweight
single-object tracking framework that integrates Transformer
and Mamba-based spatiotemporal modeling. First, we propose
the target-aware feature purification preprocessor (TAFPP) ,
designed to dynamically enhance target representation through
a synergistic combination of the dynamic position acuity module
(DPAM) and spectral channel recalibrator (SCR). Second, we
introduce the recurrent Mamba interaction pyramid (RM-IP)
to replace traditional recurrent neural network-based struc
tures, leveraging a state-space model for efficient and expressive
temporal modeling with significantly reduced parameter over
head. Finally, we propose the elastic reconstructive multi-scale
fusion (ERMSF) module, which adopts a four-branch parallel
architecture to achieve effective multiscale feature fusion and
dynamic shape adaptation, thereby enhancing robustness against
target deformations and scale variations. Extensive experiments
conducted on benchmark datasets, including LaSOT, Track
ingNet, and GOT-10k, demonstrate the effectiveness of TSTrack.
The results show that TSTrack achieves a superior tracking
accuracy while maintaining a lightweight design, significantly
outperforming existing state-of-the-art methods.

### Simple architecture

SUTrack unifies different modalities into a unified representation and trains a Transformer encoder.

![SUTrack_framework](framework.png)
