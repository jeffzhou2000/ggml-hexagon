### Compliance Statement

we should strictly follow Qualcomm's IPR policy, even in open-source community.


### the [KanTV](https://github.com/kantv-ai) way

- Simple is the beautiful

  we believe the philosophy of "<b>simple is beautiful</b>" which <b>comes from the great Unix</b>.

- Make it run, then make it right, then make it fast

- Explore and have fun!

  we believe the philosophy of <b>try crazy ideas, build wild demos, and push the edge of what’s possible</b>(which is one of the core spirits of ggml-way).

- The rule-based order

  we respect the rule-based order and we respect the IPR.

### README

- QNN_SDK: a customized/tailored Qualcomm's QNN SDK for build project ggml-hexagon conveniently. the fully QNN SDK could be found at Qualcomm's offcial website: https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk

- Hexagon_SDK: a customized/tailored Qualcomm's Hexagon SDK for build project ggml-hexagon conveniently. the fully Hexagon SDK could be found at Qualcomm's offcial website: https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools. one more important thing, the fully Hexagon SDK must be obtained with a Qualcomm Developer Account.

- [ggml-dsp](https://github.com/zhouwg/ggml-hexagon/tree/self-build/prebuilts/ggml-dsp): prebuilt libggmldsp-skel.so for Qualcomm Hexagon NPU on Android phone equipped with Qualcomm Snapdragon <b>high-end</b> mobile SoC

- models: customized LLM models for compare on-device inference peformance between QNN-CPU, QNN-GPU, QNN-NPU, cDSP, the default ggml in this project

   - t5-very-small-random-F32.gguf : original author https://huggingface.co/stas/t5-very-small-random
