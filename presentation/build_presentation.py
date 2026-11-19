"""Build the project overview from completed experiment artifacts.

Run with python-pptx and matplotlib available. Does not modify experiment files.
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ASSETS = OUT/'assets'
ASSETS.mkdir(exist_ok=True)
BG='101C2D'; CARD='1C2B40'; WHITE='F3F6FA'; MUTED='AEBFD2'; TEAL='53D2BC'; ORANGE='FFB466'; BLUE='78AEFF'

def read(path): return json.loads((ROOT/path).read_text())
scifact = read('results/scifact-heldout/results.json')
fiqa = read('results/fiqa-run/test/results.json')
generation = read('results/generation-algorithms-3b-256/results.json')
intervals = read('results/analysis/confidence-intervals.json')
status = read('results/generation-heldout-3b-256/status.json')
stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%B %d, %Y • %I:%M %p ET')
(OUT/'status-snapshot.json').write_text(json.dumps({'captured_at':stamp,'heldout_status':status},indent=2))
with (ROOT/'results/fiqa-scaling/summary.csv').open() as f: scaling=list(csv.DictReader(f))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':'#'+WHITE,
                     'axes.labelcolor':'#'+WHITE,'xtick.color':'#'+MUTED,'ytick.color':'#'+MUTED,
                     'axes.edgecolor':'#'+MUTED,'axes.facecolor':'#'+BG,'figure.facecolor':'#'+BG})
colors={'flat':'#'+MUTED,'hnsw':'#'+TEAL,'ivf_flat':'#'+ORANGE}
names={'flat':'Exact','hnsw':'HNSW','ivf_flat':'IVF-Flat'}
fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
sizes=[5000,20000,57638]
for method in colors:
    values=[[float(r['median_ms']) for r in scaling if r['method']==method and int(r['corpus_size'])==size] for size in sizes]
    means=np.array([np.mean(v) for v in values])
    ax.errorbar(sizes,means,yerr=[means-[min(v) for v in values],[max(v) for v in values]-means],marker='o',lw=2.5,capsize=5,color=colors[method],label=names[method])
ax.set_yscale('log'); ax.set_xticks(sizes,['5,000','20,000','57,638'])
ax.set_xlabel('Corpus documents'); ax.set_ylabel('Median search latency (ms, log scale)')
ax.grid(alpha=.18); ax.legend(facecolor='#'+CARD,labelcolor='#'+WHITE)
fig.savefig(ASSETS/'scaling.png',dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
for offset,method in [(-.12,'hnsw'),(.12,'ivf_flat')]:
    cis=[intervals[d][method]['evidence_recall_minus_exact'] for d in ['scifact','fiqa']]
    mean=np.array([c['mean']*100 for c in cis]); low=np.array([c['low']*100 for c in cis]); high=np.array([c['high']*100 for c in cis])
    ax.errorbar(mean,np.array([1,0])+offset,xerr=[mean-low,high-mean],fmt='o',markersize=8,capsize=6,color=colors[method],label=names[method])
ax.set_yticks([1,0],['SciFact','FiQA']); ax.set_ylim(-.5,1.5)
ax.axvline(0,color='#'+MUTED,ls='--'); ax.set_xlabel('Evidence recall difference vs exact (percentage points)')
ax.legend(facecolor='#'+CARD,labelcolor='#'+WHITE); ax.grid(axis='x',alpha=.18)
fig.savefig(ASSETS/'intervals.png',dpi=180); plt.close(fig)

prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
notes=[]
def box(slide,x,y,w,h,fill=CARD):
    shape=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb=RGBColor.from_string(fill); shape.line.fill.background()
    return shape
def text(slide,x,y,w,h,value,size=22,color=WHITE,bold=False):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    frame=shape.text_frame; frame.word_wrap=True
    frame.margin_left=frame.margin_right=Inches(.02)
    frame.margin_top=frame.margin_bottom=0
    for i,line in enumerate(value.split('\n')):
        p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
        p.text=line; p.font.name='Aptos'; p.font.size=Pt(size); p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
        p.space_after=Pt(12)
    return shape
def slide(title,section,source,note):
    s=prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string(BG)
    text(s,.55,.28,12,.3,section.upper(),12,TEAL,True)
    text(s,.55,.85,12.25,.85,title,30,WHITE,True)
    box(s,.55,6.93,12.2,.015,TEAL)
    text(s,.55,7.08,11.5,.25,source,10,MUTED)
    text(s,12.25,7.03,.5,.3,f'{len(prs.slides):02}',13,MUTED)
    s.notes_slide.notes_text_frame.text=note+'\n\nSource: '+source
    notes.append((title,note,source)); return s
def card(s,x,y,w,title,body):
    box(s,x,y,w,3.5)
    text(s,x+.22,y+.22,w-.44,.65,title,23,TEAL,True)
    text(s,x+.22,y+1.0,w-.44,2.25,body,21)
def banner(s,value):
    box(s,.6,5.95,12.1,.68)
    text(s,.82,6.09,11.65,.43,value,18,TEAL)
def table(s,headers,rows,widths,y=2.0,row_h=.58):
    x=.65
    for header,w in zip(headers,widths):
        box(s,x,y,w-.03,row_h,TEAL); text(s,x+.12,y+.12,w-.23,row_h-.12,header,17,BG,True);x+=w
    for n,row in enumerate(rows):
        x=.65
        for item,w in zip(row,widths):
            box(s,x,y+(n+1)*row_h,w-.03,row_h-.025,CARD)
            text(s,x+.12,y+(n+1)*row_h+.12,w-.23,row_h-.12,str(item),18); x+=w

s=slide('Search algorithms in retrieval-augmented generation','Project overview','Project snapshot • '+stamp,
        'Present this as a controlled empirical algorithms project. The study follows retrieval speed and quality into a limited scientific-claim generation task. The final held-out generation evaluation is still running at this snapshot; no partial accuracy is presented.')
text(s,.65,2.0,10.8,1.25,'When does faster retrieval\nmake a better RAG system?',38,TEAL,True)
text(s,.65,3.7,11.8,1,'Exact search • HNSW • IVF-Flat\nSciFact + FiQA | controlled scaling | generation evaluation',23)
text(s,.65,5.35,10,.5,'Aidan Basloe & Josh Pomeroy',23,MUTED)

s=slide('The research question','01 / Motivation','docs/project-plan.md; docs/GENERATION_FINAL_PROTOCOL.md',
        'Separate the retrieval question from the downstream question. Better means a measured tradeoff, not an assumption that approximate methods improve answers. End-to-end generation claims must wait for the frozen held-out evaluation.')
card(s,.65,2,3.9,'Speed','How much search time\ncan approximation save?')
card(s,4.72,2,3.9,'Quality','Which relevant documents\nand final verdicts change?')
card(s,8.79,2,3.9,'Practical value','Do the gains justify\nbuild cost and storage?')
banner(s,'Hold the model and documents fixed; change the search algorithm.')

s=slide('Where the search algorithm fits','02 / System','benchmark.py; generation_pilot.py; generation_heldout.py',
        'Corpus documents are embedded once with normalized BGE vectors. The query uses the same encoder. Search produces document IDs, then a common packing policy constructs the prompt. Query encoding, search, packing, and generation are distinct costs. The final held-out runner measures these online.')
for x,label in [(0.65,'Query\nembedding'),(3.18,'Vector\nsearch'),(5.71,'Context\npacking'),(8.24,'LLM\ngeneration'),(10.77,'Verdict +\ncitations')]:
    box(s,x,2.35,2.0,1.35,TEAL if x==3.18 else CARD)
    text(s,x+.12,2.65,1.76,.9,label,21,BG if x==3.18 else WHITE,True)
    if x<10:text(s,x+2.05,2.77,.45,.45,'→',26,TEAL)
text(s,.7,4.3,11.8,1.1,'Indexed corpus → fixed BGE-small embeddings → same vectors for all methods\nSearch quality, context retention, and answer quality are separate measurements.',23)

s=slide('Three search methods, one comparison','03 / Algorithms','benchmark.py; FAISS index documentation',
        'Exact inner-product search is the reference for ANN recall, not a perfect relevance oracle. HNSW searches a graph. IVF-Flat clusters vectors and searches selected lists while retaining full-precision vectors. Graph and cluster construction add cost. This project does not implement LSH, BK-trees, or product quantization.')
card(s,.65,2,3.9,'Exact / Flat','Scan every vector.\nExact top-k reference.\nMinimal index construction.')
card(s,4.72,2,3.9,'HNSW','Navigate a neighbor graph.\nTune efSearch.\nExtra build cost and storage.')
card(s,8.79,2,3.9,'IVF-Flat','Search selected clusters.\nTune nprobe.\nTrain cluster assignments.')
banner(s,'Normalized vectors + inner product + identical document IDs and k=10.')

s=slide('The project progressed in four stages','04 / Process','Repository experiment artifacts and protocol documents',
        'First validate the small dataset, then expand to a larger domain. Next assess uncertainty and scale within a single dataset. Finally investigate generated claim verification, including failures. Caching and resumable background workers allowed longer experiments to run reliably.')
for i,(a,b) in enumerate([('1  Establish','SciFact pilot; metric checks; fixed embeddings'),('2  Compare','FiQA development tuning and test evaluation'),('3  Validate','Repeated builds, confidence intervals, scaling, truncation'),('4  Connect','Generation pilots → failure review → frozen held-out run')]):
    box(s,.65,1.95+i*1.02,12,.87)
    text(s,.86,2.13+i*1.02,2.45,.45,a,22,TEAL,True)
    text(s,3.35,2.13+i*1.02,9,.5,b,21)

s=slide('A controlled experimental design','05 / Data & controls','SciFact/FiQA results metadata; docs/FIQA_RUN.md',
        'SciFact uses a reproducible subdivision of the BEIR test queries: 60 for development and 240 held out. FiQA uses its official 500-query development and 648-query test splits. All methods share cached BGE-small-en-v1.5 embeddings and a pinned revision. Settings are selected on development performance and then fixed for test evaluation.')
table(s,['Dataset','Documents','Development','Test / held-out'],[['SciFact','5,183','60','240'],['FiQA','57,638','500','648']],[3.2,2.8,2.8,3.2])
text(s,.8,4.2,11.8,1.2,'Fixed: embedding revision, corpus, IDs, similarity objective, k and hardware.\nTune search effort on development queries; do not tune on test outcomes.',23)
banner(s,'CPU search • single-threaded FAISS • reusable embeddings • recorded configurations')

s=slide('What each metric tells us','06 / Measurement','benchmark.py; results/analysis/report.md',
        'ANN recall measures overlap with exact neighbors. Evidence recall is recall against positive dataset relevance judgments, which are not proof of answer support. NDCG accounts for rank. Serialized size is not peak resident memory. Generation correctness and citation support require additional evaluation.')
table(s,['Metric','Question answered'],[['ANN recall@10','Does approximation reproduce exact neighbors?'],['Evidence recall@10 / NDCG@10','Are judged-relevant documents retrieved and ranked well?'],['Median / p95 latency','How fast are typical and slower searches?'],['Build time / serialized bytes','What does the index cost to create and store?'],['Verdict accuracy / total latency','Does the full system produce a correct, timely response?']],[4.1,7.9],row_h=.64)

s=slide('Initial retrieval results: fast search, small quality losses','07 / Completed findings','results/scifact-heldout/results.json; results/fiqa-run/test/results.json',
        'These are the original held-out/test measurements, not averages from the later scaling study. HNSW settings are 32 for SciFact and 64 for FiQA; IVF nprobe is 16 for both. HNSW is about 10x and 50x faster than exact search respectively. Absolute latency savings are far smaller than these ratios suggest.')
rows=[]
for dataset,result in [('SciFact',scifact),('FiQA',fiqa)]:
    for r in result['summary']:rows.append([dataset,names[r['method']],f"{r['median_ms']:.3f}",f"{100*r['ann_recall']:.2f}%",f"{100*r['evidence_recall']:.2f}%"])
table(s,['Dataset','Method','Median ms','ANN recall','Evidence recall'],rows,[2.1,2.3,2.3,2.5,2.8],y=1.85,row_h=.5)
banner(s,'FiQA: ~50× faster search saves about 4.2 milliseconds per query—not 50× faster RAG.')

s=slide('We tested stability, not just one favorable run','08 / Reliability','results/analysis/confidence-intervals.json; results/analysis/report.md',
        'Paired percentile bootstrap intervals use 5,000 resamples of queries from the original test runs. They condition on the built index. Five separate construction seeds and timing orders assess build/run variation separately. The HNSW evidence differences include zero; this is not proof of equivalence. IVF intervals here are below zero and describe a small observed loss.')
s.shapes.add_picture(str(ASSETS/'intervals.png'),Inches(.65),Inches(1.9),width=Inches(7.4))
text(s,9.0,2.15,3.6,2.9,'5 independent builds\n\n5,000 paired bootstrap samples\n\n95% query-level intervals',23)
banner(s,'Query uncertainty and build-to-build variation are different sources of uncertainty.')

s=slide('Controlled scaling isolates corpus growth','09 / Scaling study','docs/SCALING_RUN.md; results/fiqa-scaling/subsets.json',
        'Use nested subsets at 5,000, 20,000 and all 57,638 FiQA documents. Keep 500 development and 648 test queries fixed. Retain all 2,944 documents judged relevant to either split and add seeded distractors. Test judgments determine membership only; search tuning uses development queries. Smaller subsets are relevance-enriched; one subset sequence cannot estimate sampling variability.')
card(s,.65,2,3.9,'5,000','Same queries.\nAll relevant documents.\nSeeded distractors.')
card(s,4.72,2,3.9,'20,000','Nested membership.\nSame embedding vectors.\nSame index structure.')
card(s,8.79,2,3.9,'57,638','Full FiQA corpus.\nFive test builds per size.\nDevelopment-selected effort.')
banner(s,'45 test method measurements = 3 sizes × 3 methods × 5 builds.')

s=slide('The search advantage grows with corpus size','10 / Scaling results','results/fiqa-scaling/summary.csv; means and min–max ranges across five builds',
        'The y axis is logarithmic. The chart uses means of per-run median latencies; error bars show observed min-max ranges, not confidence intervals. HNSW speedups are approximately 11.4x, 17.8x and 50.7x. Search effort is tuned separately by size, so these are practical configurations near a recall target, not a pure asymptotic-complexity measurement.')
s.shapes.add_picture(str(ASSETS/'scaling.png'),Inches(.65),Inches(1.9),width=Inches(7.4))
text(s,9.05,2.05,3.55,.65,'11.4× → 50.7×',28,TEAL,True)
text(s,9.05,2.95,3.55,2.4,'HNSW speedup as corpus grows\n\nMean ANN recall stays above 96% at all three sizes.',22)
banner(s,'HNSW construction also rises: about 0.6 seconds → 19.5 seconds.')

s=slide('Index choice depends on the workload','11 / Interpreting efficiency','results/fiqa-scaling/report.md; derived illustration using mean build times and latencies',
        'Illustrative break-even uses mean build times and mean per-run median latencies at full FiQA size: (19.5430-.0429)/((4.5045-.0889)/1000), approximately 4,400 queries. This is a simplified query-work model, not measured total throughput. Shared embeddings, update costs, concurrency and generation are excluded. Serialized storage is distinct from RAM.')
text(s,.8,2.0,11.7,.7,'Total index work ≈ build time + number of queries × search time',27,TEAL,True)
card(s,.65,3.0,5.9,'Many searches','Construction can be amortized.\nHNSW’s low search latency becomes valuable.')
card(s,6.75,3.0,5.9,'Few searches / frequent rebuilds','Exact search may remain practical.\nFiQA illustration: ~4,400 queries to recover HNSW’s extra build cost.')

s=slide('Generation connects retrieval to the final response','12 / Downstream experiment','docs/GENERATION_PILOT.md; docs/GENERATION_ALGORITHM_COMPARISON.md',
        'This is generated scientific claim verification, not open-ended general question answering. Original SciFact labels provide support, refutation and insufficient-evidence references. The pilot uses 24 balanced development claims. Begin with a small 0.5B generator for feasibility, then use pinned Qwen2.5-3B-Instruct. No-context is an evidence-only abstention baseline; gold evidence is privileged and can still be truncated.')
text(s,.7,1.9,11.9,.65,'Task: SUPPORTED / REFUTED / INSUFFICIENT EVIDENCE + explanation + citations',23,TEAL,True)
table(s,['Condition','Role'],[['No context','Check evidence-only abstention'],['Exact / HNSW / IVF-Flat','Compare the passages produced by each index'],['Gold evidence documents','Diagnose retrieval vs context/model failures']],[4.3,7.7],y=2.85,row_h=.63)
text(s,.8,5.65,11.7,.8,'Fixed prompt, packing policy and greedy decoding; every output is checkpointed.',21)

s=slide('A pilot exposed an output-budget problem','13 / What changed during development','results/generation-comparison-3b/summary.json; generation-comparison-3b-256/summary.json',
        'The three-condition 3B experiment replayed identical prompts at 96 and 256 output tokens. Seventeen exact-context outputs reached 96 tokens, but one was already valid; 16 invalid outputs were recovered by increasing the cap. No 256-token response hit the cap. The parser and prompts were kept unchanged to isolate the budget effect. Higher accuracy does not imply improved model reasoning: many previously incomplete outputs simply became scoreable.')
table(s,['Exact-context metric','96-token cap','256-token cap'],[['Valid response schema','29.2%','95.8%'],['Verdict accuracy','20.8%','58.3%'],['Median generation time','112 s','131 s']],[5.4,3.3,3.3],row_h=.7)
banner(s,'Keep raw outputs: formatting failures can masquerade as reasoning failures.')

s=slide('Development generation results: same verdicts','14 / Completed pilot—not final evaluation','results/generation-algorithms-3b-256/results.json; 24 claims per condition',
        'Strict accuracy is shown. All three retrieval methods produce identical verdicts on every development claim. HNSW packed prompts match exact in 21/24 cases, IVF in 20/24. All outputs stop naturally under the 256-token cap. Median timings are generation-only saved-prompt replays; roughly one-second differences are not evidence of algorithmic speedup. Normalized citation formatting retains strict scores separately.')
rows=[]
gnames={'no_context':'No context','flat':'Exact','hnsw':'HNSW','ivf_flat':'IVF-Flat','gold_evidence':'Gold evidence'}
for r in generation['summary']: rows.append([gnames[r['condition']],f"{r['accuracy']*100:.1f}%",f"{r['median_generation_ms']/1000:.1f} s"])
table(s,['Context','Verdict accuracy','Median generation time'],rows,[4.5,3.5,4],y=1.9,row_h=.55)
banner(s,'Same 24 verdicts ≠ general equivalence. Most packed prompts were identical.')

s=slide('Failure analysis revealed a context bottleneck','15 / Lessons from review','docs/GENERATION_FINAL_REVIEW.md; unblinded assistant audit',
        'The review covered incorrect development responses and all gold-evidence responses. It is explicitly an unblinded assistant audit, not completed human faithfulness scoring. Under the 256-token-per-document policy, only 7/16 evidence-bearing gold contexts retained one complete annotated rationale set by conservative exact-sentence matching. Partial text can still be informative. Distinguish this generator-context issue from embedding truncation, measured at 8.8% SciFact and 4.2% FiQA documents over the BGE limit.')
card(s,.65,2,3.9,'Context can lose evidence','Only 7/16 evidence-bearing gold contexts retained a complete annotated rationale set.')
card(s,4.72,2,3.9,'Models can misread evidence','Some failures reverse the direction of visible findings or overlook differences.')
card(s,8.79,2,3.9,'Correct verdict ≠ support','A citation ID can be valid while the explanation is not supported by its passage.')
banner(s,'“Gold evidence documents” are not a guaranteed upper bound after truncation.')

s=slide('Final evaluation: frozen and still running','16 / Current status','docs/GENERATION_FINAL_PROTOCOL.md; results/generation-heldout-3b-256/status.json',
        'Status is a snapshot, not a live indicator. Do not present partial held-out accuracy or retune from partial results. The final protocol evaluates all 240 reserved claims with their natural class distribution. It measures online query embedding, search, context packing, generation and direct end-to-end timing. The held-out partition was already used for retrieval evaluation but not for generation selection. Human faithfulness review remains unfinished.')
text(s,.8,1.95,11.7,.7,f"{status.get('completed',0):,} / {status.get('total',1200):,} outputs recorded",34,TEAL,True)
text(s,.8,2.85,11.7,.5,'Snapshot: '+stamp,18,MUTED)
text(s,.8,3.65,11.6,1.8,'240 held-out claims × 5 conditions\nFrozen model, prompt, token budgets, index settings and scoring\nLive end-to-end latency + paired accuracy intervals at completion',24)
banner(s,'Final generation findings are pending. No partial held-out performance is shown.')

s=slide('What the project can—and cannot—claim','17 / Conclusions & next steps','Completed retrieval/scaling results; development generation study; frozen final protocol',
        'This is a focused controlled course project, not a universal RAG leaderboard. Scope is one embedding model, one final generator, one machine and scientific claim verification for generation. Complete the frozen run, inspect errors and conduct a documented human review. Peak resident-memory measurement and broad generalization are not established. The project already has sufficient breadth; prioritize interpretation and reproducibility over adding unrelated algorithms.')
card(s,.65,2,5.9,'Supported now','Approximate search reduces retrieval cost.\nIts advantage grows in this scaling study.\nDevelopment verdicts were preserved.')
card(s,6.75,2,5.9,'Still to establish','Held-out answer quality and total latency.\nHuman-checked explanation support.\nGeneralization beyond this setup.')
banner(s,'The contribution: explain when retrieval efficiency matters—and where later stages limit it.')

s=slide('Sources and reproducible artifacts','18 / Reference map','Local repository artifacts; primary model/dataset documentation',
        'All numeric findings come from this project’s saved artifacts. Paths are relative to the repository root. The PowerPoint includes these speaker notes, and the companion Markdown file contains the full note text. Charts are regenerated from numeric result files by build_presentation.py. The status snapshot is saved separately.')
text(s,.8,1.9,11.9,3.0,'Retrieval: results/scifact-heldout/ and results/fiqa-run/test/\nReliability: results/analysis/report.md\nScaling: results/fiqa-scaling/report.md and summary.csv\nGeneration: results/generation-algorithms-3b-256/\nReview & frozen protocol: docs/GENERATION_FINAL_REVIEW.md and GENERATION_FINAL_PROTOCOL.md',19)
text(s,.8,5.0,11.9,1.25,'BEIR: github.com/beir-cellar/beir  |  SciFact: github.com/allenai/scifact\nFAISS: github.com/facebookresearch/faiss\nModels: huggingface.co/BAAI/bge-small-en-v1.5  |  huggingface.co/Qwen/Qwen2.5-3B-Instruct',16,MUTED)

prs.save(OUT/'RAG_Project_Overview.pptx')
(OUT/'speaker-notes.md').write_text('# Project overview — speaker notes\n\nSnapshot: '+stamp+'\n\n'+'\n\n'.join(f'## {i}. {title}\n\n{note}\n\nSource: `{source}`' for i,(title,note,source) in enumerate(notes,1)))
print(f'Created {len(prs.slides)} slides: {OUT / "RAG_Project_Overview.pptx"}')
