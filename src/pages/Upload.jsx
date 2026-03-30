import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import ReactMarkdown from 'react-markdown'

const API_URL = 'https://fhkhthjkkl-medpanel.hf.space'

const SCAN_TYPES = [
  { value: 'xray', label: 'Chest X-Ray', icon: '🫁', desc: 'Pneumonia, effusion, cardiomegaly' },
  { value: 'skin', label: 'Skin Lesion', icon: '🔬', desc: 'Melanoma, BCC, actinic keratosis' },
  { value: 'retinal', label: 'Retinal Scan', icon: '👁️', desc: 'Diabetic retinopathy grading' },
]

const QUESTIONS = {
  xray: [
    { id: 'symptoms', label: 'Current symptoms', placeholder: 'e.g. fever, cough, shortness of breath' },
    { id: 'duration', label: 'Duration of symptoms', placeholder: 'e.g. 3 days, 2 weeks' },
    { id: 'history', label: 'Relevant medical history', placeholder: 'e.g. smoker, previous TB, heart condition' },
  ],
  skin: [
    { id: 'body_location', label: 'Location of lesion on body', placeholder: 'e.g. lower lip, forearm, back, lower leg' },
    { id: 'duration', label: 'How long has it been present?', placeholder: 'e.g. 2 months, appeared recently' },
    { id: 'changes', label: 'Any recent changes?', placeholder: 'e.g. growing, color change, started bleeding' },
    { id: 'sun_exposure', label: 'Sun exposure history', placeholder: 'e.g. works outdoors, history of sunburn' },
  ],
  retinal: [
    { id: 'diabetes_type', label: 'Type of diabetes', placeholder: 'e.g. Type 1, Type 2, gestational' },
    { id: 'diabetes_duration', label: 'Duration of diabetes', placeholder: 'e.g. 5 years, recently diagnosed' },
    { id: 'hba1c', label: 'Last HbA1c level (if known)', placeholder: 'e.g. 7.2%, unknown' },
    { id: 'last_eye_exam', label: 'Last eye examination', placeholder: 'e.g. 1 year ago, never' },
  ],
}

const RISK_COLORS = {
  Low: 'risk-low',
  Medium: 'risk-medium',
  High: 'risk-high',
}

export default function Upload() {
  const [step, setStep] = useState(1)
  const [patientName, setPatientName] = useState('')
  const [scanType, setScanType] = useState('')
  const [answers, setAnswers] = useState({})
  const [image, setImage] = useState(null)
  const [imageFile, setImageFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const onDrop = useCallback(files => {
    const file = files[0]
    if (file) {
      setImageFile(file)
      setImage(URL.createObjectURL(file))
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png'] },
    maxFiles: 1
  })

  const handleAnalyze = async () => {
    if (!imageFile || !patientName) return
    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', imageFile)
      formData.append('patient_name', patientName)
      formData.append('scan_hint', scanType)
      if (answers.body_location) formData.append('body_location', answers.body_location)

      // Build additional context from answers
      const contextParts = []
      Object.entries(answers).forEach(([key, val]) => {
        if (val && key !== 'body_location') {
          contextParts.push(`${key.replace(/_/g, ' ')}: ${val}`)
        }
      })

      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error(`API error: ${response.status}`)
      const data = await response.json()
      setResult(data)
      setStep(4)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setStep(1)
    setPatientName('')
    setScanType('')
    setAnswers({})
    setImage(null)
    setImageFile(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-12">

      {/* Header */}
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-white mb-2" style={{ fontFamily: 'Syne, sans-serif' }}>
          New Scan Analysis
        </h1>
        <p className="text-slate-400 text-sm">Complete the intake form for accurate AI-assisted triage</p>
      </div>

      {/* Progress */}
      {step < 4 && (
        <div className="flex items-center gap-2 mb-10">
          {[1, 2, 3].map(s => (
            <div key={s} className="flex items-center gap-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-mono font-bold transition-all duration-300 ${
                s < step ? 'bg-cyan-500 text-navy-900' :
                s === step ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40' :
                'bg-slate-800 text-slate-600 border border-slate-700'
              }`} style={{ color: s < step ? '#060B18' : undefined }}>
                {s < step ? '✓' : s}
              </div>
              {s < 3 && <div className={`h-px w-16 transition-all duration-300 ${s < step ? 'bg-cyan-500' : 'bg-slate-700'}`}></div>}
            </div>
          ))}
          <span className="ml-4 text-slate-500 text-xs font-mono">
            {step === 1 ? 'Patient Info' : step === 2 ? 'Clinical Questions' : 'Upload Image'}
          </span>
        </div>
      )}

      {/* Step 1: Patient Info + Scan Type */}
      {step === 1 && (
        <div className="animate-slide-up space-y-6">
          <div>
            <label className="block text-slate-400 text-xs font-mono mb-2 tracking-wider">PATIENT NAME</label>
            <input
              type="text"
              value={patientName}
              onChange={e => setPatientName(e.target.value)}
              placeholder="Enter patient full name..."
              className="w-full bg-navy-800/60 border border-slate-700 hover:border-slate-600 focus:border-cyan-500/50 text-white placeholder-slate-600 rounded-xl px-4 py-3 outline-none transition-all duration-200 text-sm"
            />
          </div>

          <div>
            <label className="block text-slate-400 text-xs font-mono mb-3 tracking-wider">SCAN TYPE</label>
            <div className="grid gap-3">
              {SCAN_TYPES.map(type => (
                <button
                  key={type.value}
                  onClick={() => setScanType(type.value)}
                  className={`p-4 rounded-xl border text-left transition-all duration-200 ${
                    scanType === type.value
                      ? 'bg-cyan-500/10 border-cyan-500/40 glow-cyan'
                      : 'bg-navy-800/40 border-slate-700 hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{type.icon}</span>
                    <div>
                      <div className="text-white font-medium text-sm">{type.label}</div>
                      <div className="text-slate-500 text-xs mt-0.5">{type.desc}</div>
                    </div>
                    {scanType === type.value && (
                      <div className="ml-auto w-5 h-5 rounded-full bg-cyan-500 flex items-center justify-center">
                        <span className="text-xs" style={{ color: '#060B18' }}>✓</span>
                      </div>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={() => setStep(2)}
            disabled={!patientName || !scanType}
            className="w-full py-3 rounded-xl font-semibold text-sm transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed bg-cyan-500 hover:bg-cyan-400"
            style={{ color: '#060B18' }}
          >
            Continue →
          </button>
        </div>
      )}

      {/* Step 2: Clinical Questions */}
      {step === 2 && (
        <div className="animate-slide-up space-y-5">
          <div className="p-4 rounded-xl bg-navy-800/40 border border-slate-700/50 flex items-center gap-3 mb-6">
            <span className="text-xl">{SCAN_TYPES.find(t => t.value === scanType)?.icon}</span>
            <div>
              <div className="text-white text-sm font-medium">{SCAN_TYPES.find(t => t.value === scanType)?.label}</div>
              <div className="text-slate-500 text-xs">Patient: {patientName}</div>
            </div>
          </div>

          {QUESTIONS[scanType]?.map(q => (
            <div key={q.id}>
              <label className="block text-slate-400 text-xs font-mono mb-2 tracking-wider">
                {q.label.toUpperCase()}
              </label>
              <input
                type="text"
                value={answers[q.id] || ''}
                onChange={e => setAnswers(prev => ({ ...prev, [q.id]: e.target.value }))}
                placeholder={q.placeholder}
                className="w-full bg-navy-800/60 border border-slate-700 hover:border-slate-600 focus:border-cyan-500/50 text-white placeholder-slate-600 rounded-xl px-4 py-3 outline-none transition-all duration-200 text-sm"
              />
            </div>
          ))}

          <div className="flex gap-3 pt-2">
            <button
              onClick={() => setStep(1)}
              className="flex-1 py-3 rounded-xl border border-slate-700 text-slate-400 hover:text-white hover:border-slate-600 font-medium text-sm transition-all duration-200"
            >
              ← Back
            </button>
            <button
              onClick={() => setStep(3)}
              className="flex-1 py-3 rounded-xl font-semibold text-sm transition-all duration-200 bg-cyan-500 hover:bg-cyan-400"
              style={{ color: '#060B18' }}
            >
              Continue →
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Upload Image */}
      {step === 3 && (
        <div className="animate-slide-up space-y-6">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${
              isDragActive
                ? 'border-cyan-500/60 bg-cyan-500/5'
                : image
                ? 'border-emerald-500/40 bg-emerald-500/5'
                : 'border-slate-700 hover:border-slate-600 bg-navy-800/30'
            }`}
          >
            <input {...getInputProps()} />
            {image ? (
              <div>
                <img src={image} alt="preview" className="max-h-64 mx-auto rounded-xl mb-4 object-contain" />
                <p className="text-emerald-400 text-sm font-mono">✓ Image ready for analysis</p>
                <p className="text-slate-500 text-xs mt-1">Click to replace</p>
              </div>
            ) : (
              <div>
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center">
                  <svg className="w-8 h-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <p className="text-white font-medium mb-1">Drop medical image here</p>
                <p className="text-slate-500 text-sm">or click to browse — JPG, PNG supported</p>
              </div>
            )}
          </div>

          {error && (
            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm font-mono">
              ⚠ {error}
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={() => setStep(2)}
              className="flex-1 py-3 rounded-xl border border-slate-700 text-slate-400 hover:text-white hover:border-slate-600 font-medium text-sm transition-all duration-200"
            >
              ← Back
            </button>
            <button
              onClick={handleAnalyze}
              disabled={!image || loading}
              className="flex-1 py-3 rounded-xl font-semibold text-sm transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed bg-cyan-500 hover:bg-cyan-400 flex items-center justify-center gap-2"
              style={{ color: '#060B18' }}
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-navy-900/30 border-t-navy-900 rounded-full animate-spin"></div>
                  Analyzing...
                </>
              ) : 'Run AI Diagnosis →'}
            </button>
          </div>

          {loading && (
            <div className="p-4 rounded-xl bg-navy-800/60 border border-slate-700/50 space-y-3 animate-fade-in">
              <div className="flex items-center gap-3 text-sm">
                <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></div>
                <span className="text-slate-400 font-mono text-xs">Vision Agent analyzing image...</span>
              </div>
              <div className="flex items-center gap-3 text-sm">
                <div className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" style={{ animationDelay: '0.5s' }}></div>
                <span className="text-slate-400 font-mono text-xs">RAG Agent retrieving medical literature...</span>
              </div>
              <div className="flex items-center gap-3 text-sm">
                <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" style={{ animationDelay: '1s' }}></div>
                <span className="text-slate-400 font-mono text-xs">Report Agent synthesizing clinical report...</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Step 4: Results */}
      {step === 4 && result && (
        <div className="animate-slide-up space-y-6">
          {/* Summary bar */}
          <div className="p-5 rounded-2xl bg-navy-800/60 border border-slate-700/50 flex items-center justify-between">
            <div>
              <div className="text-slate-400 text-xs font-mono mb-1">PRIMARY FINDING</div>
              <div className="text-white font-bold text-xl">{result.top_finding}</div>
              <div className="text-slate-500 text-xs mt-1">{result.scan_type?.replace('_', ' ').toUpperCase()} · {patientName}</div>
            </div>
            <div className={`px-4 py-2 rounded-lg font-mono text-sm font-bold ${RISK_COLORS[result.risk_level] || 'risk-low'}`}>
              {result.risk_level?.toUpperCase()} RISK
            </div>
          </div>

          {/* Findings */}
          {result.findings && (
            <div className="p-5 rounded-xl bg-navy-800/40 border border-slate-700/50">
              <div className="text-slate-400 text-xs font-mono mb-4 tracking-wider">CV MODEL FINDINGS</div>
              <div className="space-y-2">
                {result.findings.slice(0, 5).map((f, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className="text-slate-300 text-sm w-40 truncate">{f.condition}</div>
                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-cyan-400 transition-all duration-700"
                        style={{ width: `${(f.confidence * 100).toFixed(0)}%` }}
                      ></div>
                    </div>
                    <div className="text-slate-400 font-mono text-xs w-12 text-right">
                      {(f.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Report */}
          <div className="p-6 rounded-xl bg-navy-800/40 border border-slate-700/50">
            <div className="text-slate-400 text-xs font-mono mb-4 tracking-wider">CLINICAL REPORT</div>
            <div className="prose prose-invert prose-sm max-w-none text-slate-300 leading-relaxed">
              <ReactMarkdown>{result.report}</ReactMarkdown>
            </div>
          </div>

          <button
            onClick={reset}
            className="w-full py-3 rounded-xl border border-slate-700 hover:border-cyan-500/30 text-slate-400 hover:text-cyan-400 font-medium text-sm transition-all duration-200"
          >
            ← Run Another Scan
          </button>
        </div>
      )}
    </div>
  )
}