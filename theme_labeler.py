# theme_labeler.py

import json
from textwrap import dedent


class ThemeLabeler:
    """
    LLM-based interpretation of discovered customer themes.

    Converts precomputed theme signals (terms, noun phrases, examples)
    into structured, business-readable JSON summaries.

    This class does NOT:
    - discover themes
    - run TF-IDF / NMF
    - cluster reviews
    """

    def __init__(self, client, model="gpt-4o-mini", temperature=0.3):
        self.client = client
        self.model = model
        self.temperature = temperature

        self.system_prompt = (
            "You are an expert in customer experience analytics. "
            "You MUST return ONLY valid JSON that strictly follows the provided schema. "
            "Do not include markdown, comments, or extra text."
        )

    # ------------------------------------------------------------------
    # PROMPT BUILDER
    # ------------------------------------------------------------------

    def _build_prompt(self, cluster_id, terms, noun_phrases, examples):
        examples_text = "\n---\n".join(examples[:3])
    
        prompt = dedent(f"""
        You are an expert in customer experience analytics.
    
        You are analyzing ONE specific customer theme discovered from reviews
        of a single business.
    
        ### Context / Theme ID
        {cluster_id}
    
        ### Core Theme Terms
        These words define the scope of the theme.
        ONLY discuss aspects directly related to these terms.
        {terms}
    
        ### Supporting Noun Phrases (Contextual Clues)
        Use these only to clarify meaning.
        Do NOT introduce new topics beyond the theme scope.
        {noun_phrases[:20]}
    
        ### Representative Reviews (Evidence)
        Base ALL conclusions strictly on these examples:
        ---
        {examples_text}
    
        ### TASK
        Produce a concise, business-ready summary of THIS theme ONLY.
    
        ### OUTPUT FORMAT (STRICT JSON ONLY)
        {{
            "theme_name": "",
            "summary": "",
            "likes": [],
            "dislikes": [],
            "business_impact": ""
        }}
    
        ### IMPORTANT RULES
        - Output VALID JSON ONLY (no markdown, no extra text)
        - Theme name must be 3–5 words and theme-specific
        - Likes and dislikes MUST relate ONLY to this theme
        - If a like or dislike is NOT clearly supported by the examples,
          return an empty array
        - Do NOT infer sentiment from unrelated aspects of the business
        - Do NOT merge multiple themes into one
        - Be concrete, factual, and evidence-driven
        """).strip()
    
        return prompt


    # ------------------------------------------------------------------
    # SINGLE THEME LABELING
    # ------------------------------------------------------------------

    def label_theme(self, cluster_id, terms, noun_phrases, examples):
        """
        Generate a structured theme label for a single discovered theme.

        Returns:
            dict with keys:
            - theme_name
            - summary
            - likes
            - dislikes
            - business_impact
        """

        prompt = self._build_prompt(
            cluster_id=cluster_id,
            terms=terms,
            noun_phrases=noun_phrases,
            examples=examples,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        raw = response.choices[0].message.content

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"ThemeLabeler returned invalid JSON:\n{raw}"
            ) from e

    # ------------------------------------------------------------------
    # MULTIPLE THEMES
    # ------------------------------------------------------------------

    def label_all_themes(self, theme_inputs):
        """
        Label multiple themes in a context.

        Parameters:
            theme_inputs: list of dicts, each with:
                - theme_id
                - terms
                - noun_phrases
                - examples

        Returns:
            dict[theme_id] -> labeled theme JSON
        """

        labeled = {}

        for theme in theme_inputs:
            theme_id = theme["theme_id"]

            labeled[theme_id] = self.label_theme(
                cluster_id=theme_id,
                terms=theme["terms"],
                noun_phrases=theme["noun_phrases"],
                examples=theme["examples"],
            )

        return labeled
