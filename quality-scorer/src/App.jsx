import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import ScorerPage from './pages/ScorerPage.jsx'
import EvaluationPage from './pages/EvaluationPage.jsx'
import AboutPage from './pages/AboutPage.jsx'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<ScorerPage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="*" element={<ScorerPage />} />
      </Route>
    </Routes>
  )
}
