import { Outlet } from 'react-router-dom'
import Nav from './Nav.jsx'
import Footer from './Footer.jsx'

/**
 * Page shell — sticky nav above, page content (Outlet) in the middle, Footer below.
 * Background and text colors are inherited from the @theme tokens.
 */
export default function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <Nav />
      <main className="mx-auto w-full max-w-[1120px] flex-1 px-10">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}
