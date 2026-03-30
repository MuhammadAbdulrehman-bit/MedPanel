import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import About from './pages/About'

export default function App() {
  return (
    <div className="min-h-screen grid-bg">
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </div>
  )
}