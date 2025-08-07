'use client'

import { useState, useEffect, useRef } from 'react'
import DashboardLayout from '@/components/dashboard-layout'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface DocumentRecord {
  _id: string
  name: string
  upload_date: string
}

export default function AdminDocumentsPage() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  async function fetchDocuments() {
    setIsLoading(true)
    setError('')
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/v1/admin/documents`, { credentials: 'include' })
      if (!res.ok) throw new Error('Failed to fetch documents')
      const data = await res.json()
      setDocuments(Array.isArray(data) ? data : data.documents || [])
    } catch (err) {
      setError('Error loading documents')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

  async function handleDelete(docId: string) {
    if (!window.confirm('Are you sure you want to delete this document?')) return
    setIsLoading(true)
    setError('')
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/v1/admin/documents/${docId}`, {
        method: 'DELETE',
        credentials: 'include',
      })
      if (!res.ok) throw new Error('Delete failed')
      fetchDocuments()
    } catch (err) {
      setError('Failed to delete document')
      setIsLoading(false)
    }
  }

  async function handleUpload(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (!fileInputRef.current?.files?.[0]) return
    setUploading(true)
    setError('')
    try {
      const file = fileInputRef.current.files[0]
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/v1/admin/documents/upload`, {
        method: 'POST',
        body: formData,
        credentials: 'include',
      })
      if (!res.ok) throw new Error('Upload failed')
      fileInputRef.current.value = ''
      fetchDocuments()
    } catch {
      setError('Failed to upload document')
    } finally {
      setUploading(false)
    }
  }

  return (
    <DashboardLayout>
      <div className="p-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-6">Document Management</h1>
        <Card>
          <CardHeader>
            <CardTitle>Knowledge Base Documents</CardTitle>
          </CardHeader>
          <CardContent>
            {error && <div className="text-red-600 mb-2">{error}</div>}
            <form onSubmit={handleUpload} className="mb-6 flex gap-2 items-center">
              <input type="file" ref={fileInputRef} required className="border rounded bg-white px-2 py-1"/>
              <Button type="submit" disabled={uploading}>{uploading ? 'Uploading...' : 'Upload Document'}</Button>
            </form>
            {isLoading ? <div>Loading...</div> : (
              <ul className="space-y-2">
                {documents.length === 0 && <li className="text-gray-400">No documents found.</li>}
                {documents.map(doc => (
                  <li key={doc._id} className="flex items-center justify-between bg-gray-100 rounded p-2">
                    <div className="">
                      <span className="font-medium">{doc.name}</span>
                      <span className="ml-4 text-xs text-gray-500">{doc.upload_date ? new Date(doc.upload_date).toLocaleString() : ''}</span>
                    </div>
                    <Button type="button" size="sm" variant="destructive" onClick={() => handleDelete(doc._id)} disabled={isLoading}>Delete</Button>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}

