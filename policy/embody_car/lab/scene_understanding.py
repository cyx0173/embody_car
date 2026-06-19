from support.vlm import VLM


class SceneUnderstanding:
    def __init__(self):
        self.vlm = VLM()

    def answer(self, image, question: str) -> str:
        return self.vlm.generate(image, question)
