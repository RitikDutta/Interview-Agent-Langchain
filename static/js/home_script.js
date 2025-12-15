// Reveal on scroll
const revealElements = document.querySelectorAll('.reveal');

const revealOnScroll = () => {
    const windowHeight = window.innerHeight;
    const elementVisible = 100;

    revealElements.forEach((reveal) => {
        const elementTop = reveal.getBoundingClientRect().top;
        if (elementTop < windowHeight - elementVisible) {
            reveal.classList.add('active');
        }
    });
};

window.addEventListener('scroll', revealOnScroll);
revealOnScroll();

// Navigation Toggle Logic
const navToggle = document.querySelector('.nav-toggle');
const navMenu = document.querySelector('.hero-nav');

if (navToggle && navMenu) {
    navToggle.addEventListener('click', () => {
        navMenu.classList.toggle('open');
        navToggle.classList.toggle('open');

        // Optional: Animate icon (e.g., turn hamburger into X)
        // For now, simpler is fine.
    });

    // Close the menu after selecting a link on small screens
    navMenu.querySelectorAll('a').forEach((link) => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('open');
            navToggle.classList.remove('open');
        });
    });
}

// CTA hint reveal after load
const heroCtas = document.querySelectorAll('.hero-cta-btn');
if (heroCtas.length) {
    setTimeout(() => {
        heroCtas.forEach((btn) => btn.classList.add('cta-highlight'));
        setTimeout(() => {
            heroCtas.forEach((btn) => btn.classList.remove('cta-highlight'));
        }, 1300);
    }, 300);
}

(function () {
    // --- data model of your architecture flow ---
    const steps = [
        {
            id: "select-domain",
            title: "Select domain",
            category: "Input",
            caption: "Tell the system what kind of interview you want.",
            description:
                "You start by choosing your interview domain — for example Data Science, Backend, or Machine Learning. This lets Interview Ready load the right question universe, difficulty bands, and strategy presets for your profile.",
            tech: "UX layer · Domain presets & routing"
        },
        {
            id: "upload-resume",
            title: "Upload resume",
            category: "Input",
            caption: "You upload your CV/Resume file.",
            description:
                "Next, you upload your resume. The system accepts your file and prepares it for downstream analysis so the interview can be fully tailored to your real experience and skills.",
            tech: "File upload · PDF/text parsing"
        },
        {
            id: "resume-eda",
            title: "Resume EDA & profiling",
            category: "Analysis",
            caption: "Automatic analysis of your background.",
            description:
                "The resume goes through an EDA (Exploratory Data Analysis) style pass. We extract key signals such as skills, tools, seniority, projects, domains, and gaps to build a rich user profile.",
            tech: "LLM-based parsing · rule + pattern extraction"
        },
        {
            id: "embeddings",
            title: "Generate embeddings",
            category: "Embedding",
            caption: "Turn text into dense vectors.",
            description:
                "Important chunks from your resume and profile are converted into dense vector representations. These embeddings capture semantic meaning instead of raw keywords.",
            tech: "OpenAI text-embedding model"
        },
        {
            id: "store-dbs",
            title: "Store in Pinecone & MySQL",
            category: "Storage",
            caption: "Vector + structured storage.",
            description:
                "Interview Ready now persists your data in two places: embeddings and question vectors live in Pinecone (vector DB), while structured user profile data, scores, and history are safely stored in MySQL.",
            tech: "Pinecone vector DB · MySQL relational DB"
        },
        {
            id: "strategy-agent",
            title: "Strategy agent chooses path",
            category: "Multi-agent",
            caption: "Pick one of three interview strategies.",
            description:
                "An interview agent chooses one of three predefined strategies (for example warm-up, depth-first, or mixed). This strategy decides how aggressive the system is, what types of questions to prioritise, and how it reacts to your past answers.",
            tech: "Multi-agent controller · strategy selection"
        },
        {
            id: "rag-question",
            title: "RAG: fetch next question",
            category: "RAG",
            caption: "Semantic search over question bank.",
            description:
                "Using the selected strategy and your profile, the agent performs semantic search over the question bank stored in the vector database. It picks the most relevant next question for you in this moment.",
            tech: "RAG retrieval · Pinecone semantic search"
        },
        {
            id: "ask-answer",
            title: "Ask question & capture answer",
            category: "Interaction",
            caption: "You respond to the AI interviewer.",
            description:
                "The chosen question is asked by the AI interviewer. You answer via text or voice. The raw answer text is captured and normalised so it can be evaluated consistently by downstream expert models.",
            tech: "LLM chat layer · input capture"
        },
        {
            id: "multi-llm",
            title: "3 LLM experts evaluate",
            category: "Evaluation",
            caption: "Three perspectives, three metrics.",
            description:
                "Your answer is sent to three specialised LLM 'experts'. Each expert scores you independently across three metrics — for example technical correctness, depth of reasoning, and communication clarity.",
            tech: "Multi-LLM ensemble · metric-wise scoring"
        },
        {
            id: "scoring-loop",
            title: "EMA scoring & adaptive loop",
            category: "Feedback loop",
            caption: "Update strengths, weaknesses, and next question.",
            description:
                "The individual scores are combined using a blend of Exponential Moving Average and overall average. Strengths and weaknesses in your profile are updated in MySQL, and the agent uses this fresh state to fetch the next best question. The loop continues until the interview is complete.",
            tech: "EMA + aggregate scoring · adaptive curriculum"
        }
    ];

    // DOM references
    const flowGrid = document.getElementById("flow-grid");
    const stepCounterEl = document.getElementById("step-counter");
    const stepTotalEl = document.getElementById("step-total");
    const progressInnerEl = document.getElementById("progress-inner");
    const progressThumbEl = document.getElementById("progress-thumb");
    const progressLabelEl = document.getElementById("progress-label");

    const detailStepPillEl = document.getElementById("detail-step-pill");
    const detailTitleEl = document.getElementById("detail-title");
    const detailBodyEl = document.getElementById("detail-body");
    const detailTechEl = document.getElementById("detail-tech");

    const prevBtn = document.getElementById("prev-btn");
    const nextBtn = document.getElementById("next-btn");

    stepTotalEl.textContent = steps.length.toString();

    const nodeElements = [];
    steps.forEach((step, index) => {
        const node = document.createElement("button");
        node.type = "button";
        node.className = "flow-node";
        node.dataset.index = index.toString();
        node.innerHTML = `
        <div class="flow-node-step">
          <span class="flow-node-dot"></span>
          Step ${index + 1} · ${step.category}
        </div>
        <div class="flow-node-title">${step.title}</div>
        <div class="flow-node-caption">${step.caption}</div>
        <div class="flow-mobile-details"></div>
      `;
        node.addEventListener("click", () => {
            currentStep = index;
            renderStep();
        });
        flowGrid.appendChild(node);
        nodeElements.push(node);
    });

    let currentStep = 0;

    function renderStep() {
        const step = steps[currentStep];

        // header count
        stepCounterEl.textContent = (currentStep + 1).toString();

        // progress bar
        const fraction = (currentStep + 1) / steps.length;
        const widthPercent = Math.max(8, fraction * 100);
        progressInnerEl.style.width = `calc(${widthPercent}%)`; // Added calc for safety if needed, or just %

        // thumb position
        const barRect = progressInnerEl.parentElement.getBoundingClientRect();
        // use fraction so thumb stays aligned with logical step, not fill width
        const thumbX = fraction * barRect.width;
        // We need to update this on resize/scroll too potentially, but basic is fine.
        // Actually, the original used pixels.
        // But if initially hidden/layout reflows, barRect.width might be 0.
        // However, since it is a static page, it should be fine.
        progressThumbEl.style.left = `${thumbX}px`;

        progressLabelEl.textContent = step.caption;

        // detail pane (desktop)
        detailStepPillEl.textContent = `Step ${currentStep + 1} · ${step.category}`;
        detailTitleEl.textContent = step.title;
        detailBodyEl.textContent = step.description;

        if (step.tech && step.tech.trim().length > 0) {
            detailTechEl.textContent = step.tech;
            detailTechEl.classList.remove("hidden");
        } else {
            detailTechEl.textContent = "";
            detailTechEl.classList.add("hidden");
        }

        // highlight nodes
        nodeElements.forEach((nodeEl, index) => {
            nodeEl.classList.remove("active", "completed");

            // Clear mobile details for all nodes first (or just set empty)
            const mobileDetailEl = nodeEl.querySelector('.flow-mobile-details');
            if (mobileDetailEl) mobileDetailEl.innerHTML = "";

            if (index === currentStep) {
                nodeEl.classList.add("active");

                // POPULATE mobile details for active node
                if (mobileDetailEl) {
                    let techHtml = "";
                    if (step.tech && step.tech.trim().length > 0) {
                        techHtml = `<div class="flow-mobile-tech">${step.tech}</div>`;
                    }
                    mobileDetailEl.innerHTML = `
                        <div class="flow-mobile-desc">${step.description}</div>
                        ${techHtml}
                    `;
                }

                // Auto scroll to active node on mobile
                if (window.innerWidth < 900) {
                    // Small delay to allow content expansion to affect layout
                    setTimeout(() => {
                        nodeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                    }, 50);
                }
            } else if (index < currentStep) {
                nodeEl.classList.add("completed");
            }
        });



        // buttons
        prevBtn.disabled = currentStep === 0;
        if (currentStep === steps.length - 1) {
            nextBtn.innerHTML = `Restart <svg viewBox="0 0 24 24" aria-hidden="true"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>`;
        } else {
            nextBtn.innerHTML = `Next step <svg viewBox="0 0 24 24" aria-hidden="true"><polyline points="9 6 15 12 9 18"></polyline></svg>`;
        }
    }

    prevBtn.addEventListener("click", () => {
        if (currentStep > 0) {
            currentStep -= 1;
            renderStep();
        }
    });

    nextBtn.addEventListener("click", () => {
        if (currentStep < steps.length - 1) {
            currentStep += 1;
        } else {
            currentStep = 0;
        }
        renderStep();
    });

    // initial render
    setTimeout(renderStep, 100); // Small delay to ensure layout is settled

    // re-compute thumb position on resize
    window.addEventListener("resize", () => {
        renderStep();
    });
})();
