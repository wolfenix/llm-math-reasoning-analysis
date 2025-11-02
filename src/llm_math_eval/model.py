import torch
import transformers
import logging
from huggingface_hub import login

# Set up logger
logger = logging.getLogger(__name__)

class LLM:
    """
    A wrapper class for a Hugging Face text-generation pipeline.
    """

    def __init__(self, model_id: str, max_new_tokens: int = 256):
        """
        Initializes the LLM class and loads the model.

        Args:
            model_id: The Hugging Face model identifier.
            max_new_tokens: Default max tokens for generation.
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.llm = self.load_llm(model_id)
        
        # Generation parameters
        self.do_sample = False
        self.temperature = 1.0

    def load_llm(self, model_id: str):
        """
        Loads the text-generation model using the Transformers pipeline.

        Args:
            model_id: The model identifier from Hugging Face.

        Returns:
            A text-generation pipeline.
        """
        logger.info(f"Loading model: {model_id}")
        try:
            # Attempt to login using a token from environment or cache
            # This is often needed for gated models like Mistral
            login(token=None, add_to_git_credential=False)
            logger.info("Hugging Face login successful.")
        except ImportError:
            logger.warning("huggingface_hub.login() failed. Proceeding without explicit login.")
        except Exception as e:
            logger.warning(f"An error occurred during Hugging Face login: {e}")


        try:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            logger.info(f"Model {model_id} loaded successfully.")
            return pipeline
        except Exception as e:
            logger.critical(f"Failed to load model pipeline: {e}")
            raise

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a response from the model.

        Args:
            system_prompt: The system-level instruction.
            user_prompt: The user's query or input.

        Returns:
            The generated response content as a string.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            outputs = self.llm(
                messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                pad_token_id=self.llm.tokenizer.eos_token_id
            )
            
            # Extract the assistant's last reply
            response = outputs[0]["generated_text"][-1]['content']
            return response
            
        except Exception as e:
            logger.error(f"Error during model generation: {e}")
            return f"Error: Could not generate response. {e}"
