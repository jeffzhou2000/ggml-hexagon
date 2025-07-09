### Compliance Statement

we should strictly follow Qualcomm's IPR policy, even in open-source community.


### The ggml-hexagon way

- Simple is beautiful

  we believe the philosophy of "<b>simple is beautiful</b>" which <b>comes from the great Unix</b>.

- Make it run, then make it right, then make it fast

- Explore and have fun!

  we believe the philosophy of <b>try crazy ideas, build wild demos, and push the edge of what’s possible</b>(which is one of the core spirits of ggml-way).

- The rule-based order

  we respect the rule-based order and we respect the IPR.

### README

- QNN_SDK: the fully QNN SDK could be found at Qualcomm's offcial website: https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk, will be downloaded automatically via [build-run-ggmlhexagon-android.sh](https://github.com/zhouwg/ggml-hexagon/blob/self-build/scripts/build-run-ggmlhexagon-android.sh).

- Hexagon_SDK: a [customized/tailored Qualcomm's Hexagon SDK](https://github.com/zhouwg/toolchain/blob/main/minimal-hexagon-sdk-6.2.0.1.xz) for build project ggml-hexagon conveniently and will be downloaded automatically via build-run-ggmlhexagon-android.sh. the fully Hexagon SDK could be found at Qualcomm's offcial website: https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools. one more important thing, the fully Hexagon SDK must be obtained with a Qualcomm Developer Account.

- [ggml-dsp](https://github.com/zhouwg/ggml-hexagon/tree/self-build/prebuilts/ggml-dsp): prebuilt libggmldsp-skel.so for Qualcomm Hexagon NPU on Android phone equipped with Qualcomm Snapdragon <b>high-end</b> mobile SoC.
