# mush_engine
Project Objective  Design and build an agentic AI workflow that generates 3 high-potential viral Instagram Reels per week for your dog’s account. The system should identify real-time trending audio and formats, then produce customized reel concepts, captions, and hashtags tailored to your dog’s persona and audience.




Researcher Agent Flowchart
    A[CLI Entrypoint] -->|Calls| B[recommend_viral_dog_reels(query, dog_profile)]
    
    B --> C{Fetch Trend Context?}
    C -->|Agent| D[LangChain Agent]
    D --> E[SearchViralDogReels Tool]
    E --> F[YouTube API (fetch_trending_dog_shorts)]
    C -->|Direct Call| E

    F --> G[Summarize Trends]
    G --> H[Trend Context Text]

    H --> I[Build Prompt (BASE_IDEA_PROMPT)]
    I --> J[ChatOpenAI LLM (gpt-4o)]

    J --> K[Completion (raw JSON or text)]
    K --> L{Valid JSON?}
    L -->|Yes| M[Parse as JSON]
    L -->|No| N[Extract JSON via substring]
    N --> M

    M --> O[Validate with Pydantic (Idea model)]
    O --> P[Attach Metadata (_index, _generated_at, _warnings)]

    P --> Q[Return Clean List of Ideas]
    Q --> R[Print Ideas (if CLI)]


Script Writer AGent Flow
    A[Input Ideas (string or dict)] --> B[build_script_prompt]
    B --> C[LangChain LLM (ChatOpenAI)]
    C --> D[Raw JSON/Text Output]

    D --> E{Valid JSON?}
    E -->|No| F[Extract JSON fragment]
    E -->|Yes| G[Validate JSON Schema]

    F --> G[Validate JSON Schema]

    G --> H{Warnings?}
    H -->|Yes (if allowed)| I[Repair Prompt via LLM]
    H -->|No| J[Accept Script]

    I --> K[Validate Repaired JSON]
    K --> H

    J --> L[ScriptResult (JSON + metadata)]
    L --> M[Heuristic Scoring (compute_script_score)]
    M --> N[Diversity Bonus (hook token comparison)]
    N --> O[Rank Scripts (leaderboard)]
    O --> P[Save JSON & Leaderboard (optional)]
    O --> Q[Output Ranked Report]

