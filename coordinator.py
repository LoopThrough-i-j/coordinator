from typing import Any
import google.generativeai as genai
import faiss
import numpy as np
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field


class Text:
    NORMAL = "\033[0;37;40m"
    GREEN = "\033[1;32m"
    RED = "\033[1;31;40m"
    BLUE = "\033[94m"


class Coordinates(BaseModel):
    x: int = Field(description="X coordinate of the located item")
    y: int = Field(description="Y coordinate of the located item")


class RefinedItem(BaseModel):
    item: str = Field(
        description="The item referred to. Should be well defined or proper noun."
    )


class FaissIndex:
    def __init__(self, index_path: str, embedding_dim: int):
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query: np.ndarray, k=1):
        if self.index.ntotal > 0:
            return self.index.search(query, k)

        return None


class PromptCache:
    def __init__(self, embedding_dim=384, index_path="prompt_index.faiss"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = FaissIndex(index_path, embedding_dim)
        self.prompts = []

    def store(self, item: str):
        print("Adding to Prompt Cache")
        embedding = self.model.encode([item])[0]
        embedding = embedding / np.linalg.norm(
            embedding
        )  # Normalize for cosine similarity
        self.index.add(np.array([embedding], dtype=np.float32))
        self.prompts.append(item)
        return len(self.prompts) - 1

    def retrieve(self, item: str, insert_if_absent: bool = False) -> int | None:
        print("Searching in Prompt Cache")
        embedding = self.model.encode([item])[0].astype(np.float32)
        embedding = embedding / np.linalg.norm(
            embedding
        )  # Normalize for cosine similarity
        search_results = self.index.search(np.array([embedding]), k=1)
        if search_results:
            distances, indices = search_results
            print(
                f"_____ {self.prompts[indices[0][0]], distances[0][0]} in Prompt Cache"
            )
            if distances[0][0] > 0.9:
                print(f"Found {self.prompts[indices[0][0]]} in Prompt Cache")
                return int(indices[0][0])

        if insert_if_absent:
            return self.store(item=item)

        return None


class ImageCache:
    def __init__(self, embedding_dim=540 * 1170 * 3, index_path="image_index.faiss"):
        self.index = FaissIndex(index_path, embedding_dim)
        self.images: list[dict[str, Any]] = []
        self.response_cache: dict[tuple[int, int], Coordinates] = {}
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

    def _extract_features(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((540, 1170))
        image_array = np.array(image).astype(np.float32).flatten()
        return image_array / np.linalg.norm(image_array)

    def store(self, image_path: str):
        print("Adding to Image Cache")
        features = self._extract_features(image_path)
        self.index.add(np.array([features], dtype=np.float32))
        image = Image.open(image_path)
        text_prompt = (
            "Provide a detailed description of what is visible "
            "in the image, along with the UI components."
        )
        print(
            "Calling Gemini VLM for image description",
            text_prompt,
        )
        response = self.model.generate_content(
            [image, text_prompt],
        )
        self.images.append({"image_path": image_path, "image_desc": response.text})
        return len(self.images) - 1

    def retrieve(self, image_path: str, insert_if_absent: bool = False) -> int | None:
        print("Searching in Image Cache")
        features = self._extract_features(image_path).astype(np.float32)

        search_results = self.index.search(np.array([features]), k=1)
        if search_results:
            distances, indices = search_results
            print(
                f"_____ {self.images[indices[0][0]].get('image_path'), distances[0][0]} in Image Cache"
            )
            if distances[0][0] > 0.999:
                print(
                    f"Found {self.images[indices[0][0]].get('image_path')} in Image Cache"
                )
                return int(indices[0][0])

        if insert_if_absent:
            return self.store(image_path=image_path)

        return None


class VLM:
    def __init__(self):
        # Gemini Model
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Other utils
        self.prompt_cache = PromptCache()
        self.image_cache = ImageCache()
        self.response_cache: dict[tuple[int, int], Coordinates] = {}
        self.items = []
        self.max_num_items = 10

    def _get_response_cache(self, prompt_id: int, image_id: int) -> Coordinates:
        return self.response_cache.get((prompt_id, image_id))

    def _set_response_cache(
        self, prompt_id: int, image_id: int, coordinates: Coordinates
    ):
        self.response_cache[(prompt_id, image_id)] = coordinates

    def _get_cached_coordinated(self, item: str, image_path: str) -> Coordinates | None:
        prompt_id = self.prompt_cache.retrieve(item)
        image_id = (
            self.image_cache.retrieve(image_path) if prompt_id is not None else None
        )
        if prompt_id is not None and image_id is not None:
            return self.response_cache.get((prompt_id, image_id))

    def _load_image(self, image_path: str):
        return Image.open(image_path)

    def _draw_and_save(self, coordinates: Coordinates, image_path: str):
        image = self._load_image(image_path)
        draw = ImageDraw.Draw(image)
        radius = 25
        x, y = coordinates.x, coordinates.y
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
        image.save("output_image.png")

    def get_coordinates(self, image_path: str, item: str):
        prompt_id = self.prompt_cache.retrieve(item)
        image_id = self.image_cache.retrieve(image_path, insert_if_absent=True)
        coordinates = self.response_cache.get((prompt_id, image_id))

        if coordinates is None:
            print("Refining item")
            image_details = self.image_cache.images[image_id]["image_desc"]
            prompt = (
                "You are requested to locate a certain item on the image. "
                "Respond with the a well defined item to be clicked on. "
                "You can get an idea from the image description. "
                f"Previous requested items in order: {self.items if self.items else 'Nothing yet'}. "
                f"Current item requested for: `{item}`. "
                "(Note: Current item might reference a previous requested item)."
                f"Image Description: {image_details}. "
            )
            print("Calling Gemini for refinement", prompt)
            response = self.gemini_model.generate_content(
                contents=prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": RefinedItem,
                },
            )
            print(response.text)
            refined_item = RefinedItem.model_validate_json(response.text)
            print("Refined item generated:", refined_item)

            item = refined_item.item

        prompt_id = self.prompt_cache.retrieve(item, insert_if_absent=True)
        coordinates = self._get_cached_coordinated(item=item, image_path=image_path)

        if coordinates is not None:
            self._set_response_cache(prompt_id, image_id, coordinates)
            self._draw_and_save(coordinates=coordinates, image_path=image_path)
            return coordinates

        image = self._load_image(image_path)
        text_prompt = (
            f"Assume you want to click on {item} on the image. "
            f"Respond with the x, y coordinates for the of {item} where it can be clicked. "
            f"Coordinates should be exactly on top of the item on the image of size {image.size}. "
            "x and y coordinates should be -1 if the "
            "object is not properly recognized."
        )

        response = self.gemini_model.generate_content(
            [image, text_prompt],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": Coordinates,
            },
        )
        print("Received response from Gemini VLM")
        print(response.text)

        coordinates: Coordinates = Coordinates.model_validate_json(response.text)

        self.items.append(item)
        self._set_response_cache(prompt_id, image_id, coordinates)
        self._draw_and_save(coordinates=coordinates, image_path=image_path)
        return coordinates


class ChatInterface:
    def __init__(self):
        self.vlm = VLM()

    def chat(self):
        print("\nWelcome to the Coordinator Chat. Type 'exit' to quit.")
        while True:
            try:
                image_path = input(f"Enter image path: {Text.GREEN}")
                if image_path.lower() == "exit":
                    break
                print(Text.NORMAL, end="")
                item = input(f"Enter item to locate: {Text.GREEN}")
                if item.lower() == "exit":
                    break
                print(Text.NORMAL, end="")

                print(Text.BLUE, end="\n\n")
                coordinates = self.vlm.get_coordinates(image_path, item)
                print(Text.NORMAL, end="\n\n")

                print("Coordinates:", Text.GREEN, coordinates, Text.NORMAL)
            except Exception as e:
                print(Text.RED)
                print(f"An error occurred: {e}", Text.NORMAL, end="\n\n")


if __name__ == "__main__":
    genai.configure(api_key="")
    chat_interface = ChatInterface()
    chat_interface.chat()
