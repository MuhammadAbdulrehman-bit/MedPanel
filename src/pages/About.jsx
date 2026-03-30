const stack = [
  { category: 'Computer Vision', items: ['EfficientNetB3 (Skin · HAM10000)', 'ResNet50 (Retinal · DR Dataset)', 'TorchXRayVision DenseNet121 (X-Ray)'] },
  { category: 'Agent Framework', items: ['LangGraph (State Machine)', 'Groq API — Llama 3.1 8B', '3-Agent Pipeline'] },
  { category: 'RAG System', items: ['ChromaDB (Vector DB)', 'Sentence Transformers', 'Medical Literature Index'] },
  { category: 'Backend', items: ['FastAPI', 'Python 3.10', 'Docker'] },
  { category: 'Deployment', items: ['HuggingFace Spaces', 'Vercel (Frontend)', 'GitHub'] },
]

const models = [
  { name: 'EfficientNetB3', task: 'Skin Lesion Classification', dataset: 'HAM10000', accuracy: '92%', classes: 7 },
  { name: 'ResNet50', task: 'Diabetic Retinopathy', dataset: 'DR Gaussian 224x224', accuracy: '81%', classes: 5 },
  { name: 'DenseNet121', task: 'Chest X-Ray Pathology', dataset: 'NIH + CheXpert + MIMIC', accuracy: 'Pretrained', classes: 18 },
]

export default function About() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-12">

      <div className="mb-12">
        <h1 className="text-4xl font-bold text-white mb-3" style={{ fontFamily: 'Syne, sans-serif' }}>
          System Architecture
        </h1>
        <p className="text-slate-400 max-w-2xl">
          MediScan AI is a production-grade multi-agent medical imaging triage system built from scratch — from model training to cloud deployment.
        </p>
      </div>

      {/* Pipeline diagram */}
      <div className="mb-12 p-6 rounded-2xl bg-navy-800/40 border border-slate-700/50">
        <div className="text-slate-400 text-xs font-mono mb-6 tracking-wider">── AGENT PIPELINE</div>
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          {[
            { label: 'Medical Image', sub: 'JPG / PNG', color: 'slate' },
            { label: 'Vision Agent', sub: 'CV Model Inference', color: 'cyan' },
            { label: 'RAG Agent', sub: 'ChromaDB Retrieval', color: 'violet' },
            { label: 'Report Agent', sub: 'Groq LLM Synthesis', color: 'amber' },
            { label: 'Clinical Report', sub: 'Structured Output', color: 'emerald' },
          ].map((node, i) => (
            <div key={i} className="flex items-center gap-2 flex-shrink-0">
              <div className={`px-4 py-3 rounded-xl border text-center min-w-28 ${
                node.color === 'cyan' ? 'bg-cyan-500/10 border-cyan-500/30' :
                node.color === 'violet' ? 'bg-violet-500/10 border-violet-500/30' :
                node.color === 'amber' ? 'bg-amber-500/10 border-amber-500/30' :
                node.color === 'emerald' ? 'bg-emerald-500/10 border-emerald-500/30' :
                'bg-slate-800/60 border-slate-700'
              }`}>
                <div className="text-white text-xs font-medium">{node.label}</div>
                <div className="text-slate-500 text-xs mt-0.5 font-mono">{node.sub}</div>
              </div>
              {i < 4 && <span className="text-slate-600">→</span>}
            </div>
          ))}
        </div>
      </div>

      {/* Models */}
      <div className="mb-12">
        <h2 className="text-slate-400 text-xs font-mono tracking-widest mb-5">── CV MODELS</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {models.map((m, i) => (
            <div key={i} className="p-5 rounded-xl bg-navy-800/40 border border-slate-700/50 hover:border-slate-600/50 transition-all">
              <div className="font-mono text-cyan-400 text-sm font-bold mb-1">{m.name}</div>
              <div className="text-white text-sm font-medium mb-3">{m.task}</div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Dataset</span>
                  <span className="text-slate-300 font-mono">{m.dataset}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Val Accuracy</span>
                  <span className="text-emerald-400 font-mono font-bold">{m.accuracy}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Classes</span>
                  <span className="text-slate-300 font-mono">{m.classes}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Tech Stack */}
      <div>
        <h2 className="text-slate-400 text-xs font-mono tracking-widest mb-5">── TECH STACK</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {stack.map((s, i) => (
            <div key={i} className="p-5 rounded-xl bg-navy-800/40 border border-slate-700/50">
              <div className="text-slate-400 text-xs font-mono tracking-wider mb-3">{s.category.toUpperCase()}</div>
              <ul className="space-y-1">
                {s.items.map((item, j) => (
                  <li key={j} className="text-slate-300 text-sm flex items-center gap-2">
                    <div className="w-1 h-1 rounded-full bg-cyan-500/60"></div>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}