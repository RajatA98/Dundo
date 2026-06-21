import { Outlet } from 'react-router-dom'
import Nav from './Nav.jsx'
import Footer from './Footer.jsx'

/**
 * Page shell — nav, full-width page content (so the hero gradient can bleed
 * edge to edge), footer. Each page/section manages its own max-width.
 */
export default function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <Nav />
      <main className="w-full flex-1">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}
