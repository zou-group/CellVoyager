#!/usr/bin/env python3
"""
Simple hypothesis seeding example - the clean, minimal approach
"""

import os
from agent import AnalysisAgent

def simple_example():
    """Basic hypothesis seeding usage"""
    
    agent = AnalysisAgent(
        h5ad_path="example/covid19.h5ad",
        paper_summary_path="example/covid19_summary.txt",
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model_name="o3-mini",
        analysis_name="hypothesis_test",
        num_analyses=1,
        max_iterations=3,
        use_deepresearch_background=False
    )

    hypotheses = [
        """
        Plot the proportion of each cell type in each sample colored by the donor of origin in a grid (4 columns)
        The x axes should correspond to the ventliation or ARDS status of each patients.
        Between each of these statuses, do a two-sided Wilcoxon rank-sum test and report the p-value on the plot.
        Use n = 6, n = 4, and n = 4 biologically independent samples for Healthy, NonVent and ARDS, respectively.
        Boxplot features: minimum whisker, 25th percentile − 1.5 × inter-quartile range (IQR) or the lowest value within;
         minimum box, 25th percentile; center, median; maximum box, 75th percentile; maximum whisker, 75th percentile + 1.5 × IQR 
         or greatest value within.
        Ensure that you execute this plot exactly as specified. Do not perform any other analyses.
        Use the highest resolution for the celltype labels.
        The steps in the analysis plan should be solely focused on correctly any bugs/errors in the plot. Each step should output the plot described.
        """
    ]
    
    
    print("🌱 Running with seeded hypotheses:")
    for i, hyp in enumerate(hypotheses, 1):
        print(f"  {i}. {hyp}")
    
    # AI will create analysis plans and execute them
    agent.run(seeded_hypotheses=hypotheses)

def mixed_example():
    """Mix seeded hypotheses with AI-generated ones"""
    
    agent = AnalysisAgent(
        h5ad_path="example/covid19.h5ad", 
        paper_summary_path="example/covid19_summary.txt",
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model_name="o3-mini",
        analysis_name="mixed_test",
        num_analyses=4,  # Want 4 total analyses
        max_iterations=3
    )
    
    # Only provide 2 hypotheses - AI will generate 2 more
    hypotheses = [
        "Neutrophils show enhanced NETosis in severe COVID-19",
        "NK cells have impaired cytotoxicity pathways"
    ]
    
    print("🌱 Running mixed approach:")
    print(f"  - {len(hypotheses)} seeded hypotheses")
    print(f"  - {agent.num_analyses - len(hypotheses)} AI-generated analyses")
    
    agent.run(seeded_hypotheses=hypotheses)

def individual_phases():
    """Use individual phase control with hypothesis seeding"""
    
    agent = AnalysisAgent(
        h5ad_path="example/covid19.h5ad",
        paper_summary_path="example/covid19_summary.txt", 
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model_name="o3-mini",
        analysis_name="phase_test",
        max_iterations=3
    )
    
    # Phase 1: Generate analysis from hypothesis
    hypothesis = "Plasma cells exhibit distinct immunoglobulin signatures in recovered patients"
    
    print(f"🧠 Generating analysis plan for: {hypothesis}")
    analysis = agent.generate_idea("", analysis_idx=0, seeded_hypothesis=hypothesis)
    
    print("Generated analysis plan:")
    for i, step in enumerate(analysis['analysis_plan'], 1):
        print(f"  {i}. {step}")
    
    # Phase 2: Execute the analysis
    print("\n🚀 Executing analysis...")
    past_analyses = agent.execute_idea(analysis, "", 0)
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    print("🌱 Hypothesis Seeding Examples")
    print("=" * 40)
    
    print("\nUsage patterns:")
    print("1. Simple: agent.run(seeded_hypotheses=['hypothesis1', 'hypothesis2'])")
    print("2. Mixed: Provide fewer hypotheses than num_analyses")
    print("3. Individual: agent.generate_idea(seeded_hypothesis='...')")
    
    simple_example()

