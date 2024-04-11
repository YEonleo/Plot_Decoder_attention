# Transformer-Based Answer Generation and Visualization Tool

This project employs Transformer-based language models to generate answers to given questions and visualizes the model's attention mechanism and the probabilities of the generated tokens. Users input questions, and the model generates corresponding answers. Additionally, the tool visualizes:
- Attention scores across different layers and tokens, providing insights into how the model is focusing on various parts of the input during the answer generation process.
- Probabilities of generated tokens across different layers, highlighting the model's decision-making process in selecting specific tokens to form coherent and contextually relevant answers.

# Analysis of Generated Answers

Our tool provides a comprehensive analysis of how the Transformer-based model processes input to generate answers. Key observations from our visualizations include:

- **Initial Layers Focus on Overall Sentence Structure:** In the initial layers of the Transformer model, we observe high attention scores across the entire sentence. This suggests that the model focuses on understanding the basic structure and context of the input sentence.

- **Mid to Later Layers Concentrate on Specific Context and Start Tokens:** From the mid to later layers, there is a noticeable shift in attention scores towards specific context-related tokens and the starting tokens of the sentences. This indicates that the model begins to fine-tune its focus, paying more attention to critical parts of the input that are relevant to generating a coherent and contextually accurate answer.

These observations help in understanding the model's layered approach to processing and generating language, where initial layers capture the broad context and the deeper layers focus on specifics necessary for accurate answer generation.


# Analysis Example: "House of Anubis"

To illustrate the insights provided by our tool, let's consider the question about the television series "House of Anubis":

**Q: The Dutch-Belgian television series that "House of Anubis" was based on first aired in what year?**  
**A: 2006**

Our analysis highlights two main observations from the attention and token probability visualizations:

- **Initial Layer Attention Distribution:** In the model's initial layers, attention is broadly distributed across the entire question, indicating a general focus on understanding the overall context and structure of the sentence.

- **Focused Attention in Mid to Later Layers:** As we progress to mid and later layers, there's a clear shift in attention towards key contextual information and the starting tokens. Specifically, attention scores increase for words related to "Dutch-Belgian television series" and "first aired," showing the model's narrowing focus on elements critical to answering the question accurately.

These patterns demonstrate the model's approach to language processing: starting from a broad understanding and moving towards specific details necessary for generating precise answers.

![token_469](https://github.com/YEonleo/Plot_Decoder_attention/assets/90837906/7cc31b0a-4ea7-4ac8-b39d-25f4a0d7553d)
