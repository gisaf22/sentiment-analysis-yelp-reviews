import json
from textwrap import dedent


class ThemeLabeler:
    """
    LLM-based interpretation of discovered customer themes.

    Converts precomputed theme signals (terms, noun phrases, examples)
    into structured, business-readable summaries.
    """

    def __init__(self, client, model="gpt-4o-mini", temperature=0.4):
        self.client = client
        self.model = model
        self.temperature = temperature

        self.system_prompt = (
            "You are an expert in customer experience analytics. "
            "Return ONLY valid JSON matching the requested schema."
        )

    # ------------------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------------------

    def _build_prompt(self, cluster_id, terms, noun_phrases, examples):
        examples_text = "\n---\n".join(examples[:3])

        return dedent(f"""
        You are analyzing customer feedback that has already been grouped
        into a coherent theme for a single business.

        Your task is to clearly describe what this theme represents.

        --------------------
        THEME CONTEXT
        --------------------

        Theme ID:
        {cluster_id}

        Key Terms:
        {terms}

        Common Phrases:
        {noun_phrases[:20]}

        Example Feedback:
        ---
        {examples_text}

        --------------------
        TASK
        --------------------

        Write a concise, business-readable interpretation of this theme.

        - Use a natural, human-readable theme name
        - Focus on what customers are consistently reacting to
        - It is okay if related aspects appear, but center on the main pattern
        - Base all statements on the provided examples

        --------------------
        OUTPUT FORMAT (JSON ONLY)
        --------------------

        {{
            "theme_name": "",
            "theme_category": "",
            "summary": "",
            "likes": [],
            "dislikes": [],
            "business_impact": ""
        }}
        """).strip()

    # ------------------------------------------------------------------
    # SINGLE THEME
    # ------------------------------------------------------------------

    def label_theme(self, cluster_id, terms, noun_phrases, examples):
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
            raise ValueError(f"Invalid JSON returned:\n{raw}") from e

    # ------------------------------------------------------------------
    # MULTIPLE THEMES
    # ------------------------------------------------------------------

    def label_all_themes(self, theme_inputs):
        labeled = {}

        for theme in theme_inputs:
            labeled[theme["theme_id"]] = self.label_theme(
                cluster_id=theme["theme_id"],
                terms=theme["terms"],
                noun_phrases=theme["noun_phrases"],
                examples=theme["examples"],
            )

        return labeled
