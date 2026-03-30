import { Link } from 'react-router-dom'

const stats = [
  { label: 'CV Models', value: '3', sub: 'EfficientNetB3 · ResNet50 · DenseNet121', color: 'cyan' },
  { label: 'Scan Types', value: '3', sub: 'Chest X-Ray · Skin · Retinal', color: 'violet' },
  { label: 'RAG Chunks', value: '36', sub: 'Medical knowledge indexed', color: 'emerald' },
  { label: 'Agents', value: '3', sub: 'Vision · RAG · Report', color: 'amber' },
]

const pipeline = [
  { step: '01', title: 'Vision Agent', desc: 'CV model analyzes the medical image and extracts findings with confidence scores', icon: '👁️' },
  { step: '02', title: 'RAG Agent', desc: 'Retrieves relevant medical literature from ChromaDB using semantic search', icon: '📚' },
  { step: '03', title: 'Report Agent', desc: 'Synthesizes CV findings + medical knowledge into a structured clinical report via Groq LLM', icon: '📋' },
]

export default function Dashboard() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-12">

      {/* Hero */}
      <div className="mb-16 animate-fade-in">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-6">
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></div>
          <span className="font-mono text-xs text-cyan-400">AUTONOMOUS MEDICAL TRIAGE SYSTEM</span>
        </div>
        <h1 className="text-5xl font-bold mb-4" style={{ fontFamily: 'Syne, sans-serif' }}>
          <span className="text-white">MediScan </span>
          <span className="text-cyan-400">AI</span>
        </h1>
        <p className="text-slate-400 text-lg max-w-2xl leading-relaxed">
          A multi-agent AI system for medical image triage. Upload a chest X-ray, skin lesion, or retinal scan — three specialized agents analyze, retrieve medical context, and generate a structured clinical report.
        </p>
        <div className="flex gap-4 mt-8">
          <Link
            to="/upload"
            className="px-6 py-3 bg-cyan-500 hover:bg-cyan-400 text-navy-900 font-semibold rounded-lg transition-all duration-200 glow-cyan text-sm"
            style={{ color: '#060B18' }}
          >
            Start New Scan →
          </Link>
          <Link
            to="/about"
            className="px-6 py-3 border border-slate-700 hover:border-slate-500 text-slate-400 hover:text-slate-200 font-medium rounded-lg transition-all duration-200 text-sm"
          >
            View Architecture
          </Link>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
        {stats.map((stat, i) => (
          <div
            key={i}
            className="p-5 rounded-xl bg-navy-800/60 border border-slate-700/50 hover:border-slate-600/50 transition-all duration-200 animate-slide-up"
            style={{ animationDelay: `${i * 0.1}s` }}
          >
            <div className="font-mono text-3xl font-bold text-white mb-1">{stat.value}</div>
            <div className="text-slate-400 text-sm font-medium mb-2">{stat.label}</div>
            <div className="text-slate-600 text-xs font-mono">{stat.sub}</div>
          </div>
        ))}
      </div>

      {/* Pipeline */}
      <div className="mb-16">
        <h2 className="font-mono text-xs text-slate-500 tracking-widest mb-6">── AGENT PIPELINE</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {pipeline.map((item, i) => (
            <div key={i} className="p-6 rounded-xl bg-navy-800/40 border border-slate-700/50 hover:border-cyan-500/20 hover:bg-navy-800/60 transition-all duration-300 group">
              <div className="flex items-start justify-between mb-4">
                <span className="font-mono text-xs text-slate-600">{item.step}</span>
                <span className="text-2xl">{item.icon}</span>
              </div>
              <h3 className="font-semibold text-white mb-2 group-hover:text-cyan-400 transition-colors">{item.title}</h3>
              <p className="text-slate-500 text-sm leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* CTA */}
      <div className="p-8 rounded-2xl bg-gradient-to-r from-cyan-500/5 to-navy-700/30 border border-cyan-500/10 text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Ready to analyze a scan?</h3>
        <p className="text-slate-400 text-sm mb-6">Upload a medical image and get a structured clinical report in seconds.</p>
        <Link
          to="/upload"
          className="inline-flex px-8 py-3 bg-cyan-500 hover:bg-cyan-400 font-semibold rounded-lg transition-all duration-200 text-sm glow-cyan"
          style={{ color: '#060B18' }}
        >
          Run AI Diagnosis
        </Link>
      </div>
    </div>
  )
}