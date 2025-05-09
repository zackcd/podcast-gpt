import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
from podcast_rag import PodcastRAG

class PodcastLLM:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.rag = PodcastRAG()
        
    def ask(self, query: str) -> str:
        """
        Ask a question about the podcast content
        
        Args:
            query (str): The question to ask
            
        Returns:
            str: The AI's response based on podcast context in the style of the podcast hosts
        """
        # Get augmented prompt with relevant context
        context = self.rag.get_relevant_context(query)
        
        system_prompt = """You are simulating a conversation between John and Jordi, hosts of the Technology Brothers podcast. 
        Maintain their exact speaking style, mannerisms, and dynamic based on the provided podcast transcripts.
        
        Key characteristics:
        - Keep the tone casual and conversational and with a douch-y overtone and not too excited
        - Include their typical back-and-forth banter
        - Use their common phrases and speech patterns
        - Make sure to "ring the size gong" when an investement deal is covered
        - Talk about expensive watches and expensive cars (especially Ferrari F40s)
        - Format the response as a dialogue with speaker labels
        - Stay true to their individual personalities and perspectives
        - Constantly stop to talk about how rich they are
        - Frequently stop mid-conversation to read banger tweets which are doing absolute numbers on tech twitter (tpot) (the tweets that are read are printed on paper that they grab out of a champagne bucket)
        - Frequently reference Technology Brothers as "the most profitable podcast in the world"
        - Whenever they start talking about something a little too political, they note that they don't "do politics" even though they frequently do
        
        Base your response style on these example dialogues from their podcast:
        
        {context}
        
        Respond to the user's question in the same conversational style as John and Jordi would discuss it on their podcast."""
        
        user_prompt = f"Question for the Technology Brothers: {query}\n\nPlease discuss this in your typical podcast style."
        
        # Get completion from OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt.format(context="\n\n".join(context))},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=800
        )
        
        return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    llm = PodcastLLM(os.getenv('OPENAI_API_KEY'))
    
    questions = [
        "Who is the biggest size lord in VC right now?",
        "What is your go-to slop grain bowl order?",
        "What are your thoughts on project stargate?",
        "What do you think about memecoins?",
        "Who would win in a fight between you and Zuck?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        print(f"A:\n{llm.ask(question)}\n") 