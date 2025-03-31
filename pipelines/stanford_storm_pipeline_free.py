from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
import os
import logging

from dspy import Example
from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import OllamaClient
from knowledge_storm.rm import SearXNG
# from knowledge_storm.utils import load_api_key
import os


logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class Pipeline:
    class Valves(BaseModel):
        SEARXNG_API_URL: str = Field(default="http://localhost:5678", description="SearXNG search engine API URL")
        URL: str = Field(default="http://localhost", description="Ollama server URL")
        PORT: int = Field(default=11434, description="Ollama server port")
        MODEL: str = Field(default="gemma3:4B", description="Ollama model to use")
        OUTPUT_DIR: str = Field(default="./results/ollama", description="Output directory for articles")
        MAX_THREAD_NUM: int = Field(default=1, description="Max number of threads")

    def __init__(self):
        self.name = "Storm-Lite Research"
        self.valves = self.Valves(
            **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        )

    async def on_startup(self):
        logger.info(f"{self.name} pipeline starting up...")

    async def on_shutdown(self):
        logger.info(f"{self.name} pipeline shutting down...")

    async def on_valves_updated(self):
        logger.info("Valves updated.")

    async def inlet(self, body: dict, user: dict) -> dict:
        logger.debug("Inlet triggered.")
        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        logger.debug("Outlet triggered.")
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.info(f"User Message: \n----------\n{user_message}\n----------\n")
        try:
            # Run your article generation function
            if "###" in user_message:
                return f"Error detected: \n{user_message}"
            result = self.run_storm_pipeline(
                topic=user_message,
                searxng_api_url=self.valves.SEARXNG_API_URL,
                url=self.valves.URL,
                port=self.valves.PORT,
                model=self.valves.MODEL,
                output_dir=self.valves.OUTPUT_DIR,
                max_thread_num=self.valves.MAX_THREAD_NUM,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
                remove_duplicate=True,
            )
            return f"{result}"
        except Exception as e:
            # logger.error(f"Error in pipeline: {e}")
            return f"An error occurred while generating the article for '{user_message}'. \n\nError: {str(e)}"
        
    def run_storm_pipeline(
        self,
        topic: str,
        searxng_api_url: str = "http://localhost:5678",
        url: str = "http://localhost",
        port: int = 11434,
        model: str = "llama3:latest",
        output_dir: str = "./results/ollama",
        max_thread_num: int = 1,
        max_conv_turn: int = 3,
        max_perspective: int = 3,
        search_top_k: int = 3,
        retrieve_top_k: int = 3,
        do_research: bool = True,
        do_generate_outline: bool = True,
        do_generate_article: bool = True,
        do_polish_article: bool = True,
        remove_duplicate: bool = True,
    ) -> str:
        # Load secrets
        # load_api_key(toml_file_path="secrets.toml")

        # Set up language models
        lm_configs = STORMWikiLMConfigs()
        ollama_kwargs = {
            "model": model,
            "port": port,
            "url": url,
            "stop": ("\n\n---",),
        }

        conv_simulator_lm = OllamaClient(max_tokens=500, **ollama_kwargs)
        question_asker_lm = OllamaClient(max_tokens=500, **ollama_kwargs)
        outline_gen_lm = OllamaClient(max_tokens=400, **ollama_kwargs)
        article_gen_lm = OllamaClient(max_tokens=700, **ollama_kwargs)
        article_polish_lm = OllamaClient(max_tokens=4000, **ollama_kwargs)

        lm_configs.set_conv_simulator_lm(conv_simulator_lm)
        lm_configs.set_question_asker_lm(question_asker_lm)
        lm_configs.set_outline_gen_lm(outline_gen_lm)
        lm_configs.set_article_gen_lm(article_gen_lm)
        lm_configs.set_article_polish_lm(article_polish_lm)

        # Engine runner arguments
        engine_args = STORMWikiRunnerArguments(
            output_dir=output_dir,
            max_conv_turn=max_conv_turn,
            max_perspective=max_perspective,
            search_top_k=search_top_k,
            max_thread_num=max_thread_num,
        )

        rm = SearXNG(
            searxng_api_url=searxng_api_url,
            # searxng_api_key=os.getenv("SEARXNG_API_KEY"),
            searxng_api_key=None,
            k=search_top_k,
        )

        # Initialize runner
        runner = STORMWikiRunner(engine_args, lm_configs, rm)

        # Run generation pipeline
        try:
            polished_article = runner.run(
                topic=topic,
                do_research=do_research,
                do_generate_outline=do_generate_outline,
                do_generate_article=do_generate_article,
                do_polish_article=do_polish_article,
                remove_duplicate=remove_duplicate,
            )
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            polished_article = None
            return str(e)
        # runner.post_run()
        # runner.summary()

        # topic = topic.replace(" ", "_")

        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir, exist_ok=True)
        # topic_dir = os.path.join(output_dir, topic)
        # if not os.path.exists(topic_dir):
        #     os.makedirs(topic_dir, exist_ok=True)
            
        # final_txt_path = os.path.join(output_dir, f"{topic}/storm_gen_article.txt")
        if polished_article:
            # with open(final_txt_path, "r", encoding="utf-8") as f:
                # return f.read()
            return str(polished_article)
        return "No article found!"

