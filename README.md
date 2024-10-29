# image_caption_generation_using_GRU
I used the Flicker8k Dataset, which can be found in Dataset Folder. The Flicker8k Dataset contains 8,000 images, each accompanied by five different captions, providing a rich and diverse set of examples for training image captioning models.

To generate captions for images, I employed an encoder-decoder architecture. In this architecture, the image features are first extracted using a Convolutional Neural Network (CNN). The CNN acts as a feature extractor, capturing the essential visual details of the images. These extracted features are then used as inputs to the decoder.

The decoder is responsible for generating the textual description of the image (using ANN to predict the next word). It processes the previous text in the caption to predict the next word. Specifically, I used a Recurrent Neural Network (RNN) with Gated Recurrent Units (GRU) as the encoder. The GRU helps to manage long-term dependencies and improve the efficiency of the learning process.

To further enhance the performance, I utilized teacher forcing methods. Teacher forcing involves using the actual target word as the next input to the decoder during training, rather than using the word predicted by the model. This technique helps the model to learn more effectively by providing it with the correct context at each step during training.

By combining these techniques, I aimed to build a robust model capable of generating accurate and coherent captions for images. The use of the Flicker8k Dataset provided a solid foundation for training, while the feature extractor (CNN), encoder-decoder architecture, GRU, and teacher forcing methods ensured that the network could learn how to caption the images correctly and efficiently.

## References

1. Young, Peter, et al. "From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference over Event Descriptions." *Transactions of the Association for Computational Linguistics* 2 (2014): 67–78. [Link](https://aclanthology.org/Q14-1006/)
2. Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to Sequence Learning with Neural Networks." *Proceedings of the 27th International Conference on Neural Information Processing Systems (NIPS)*. 2014. [Link](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks)
3. Cho, Kyunghyun, et al. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. 2014. [Link](https://aclanthology.org/D14-1179/)
4. Williams, Ronald J., and David Zipser. "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks." *Neural Computation* 1, no. 2 (1989): 270–280. [Link](https://doi.org/10.1162/neco.1989.1.2.270)
5. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks." *Advances in Neural Information Processing Systems* 25 (2012): 1097-1105. [Link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)


