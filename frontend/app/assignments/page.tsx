'use client'

import DashboardLayout from '@/components/dashboard-layout'
import { Card, CardContent } from '@/components/ui/card'
import { FileText } from 'lucide-react'

export default function AssignmentsPage() {
  return (
    <DashboardLayout>
      <div className="p-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-6">Assignments</h1>
        
        <Card>
          <CardContent className="p-8 text-center">
            <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No assignments</h3>
            <p className="text-gray-500">Your assignments will appear here when they are available.</p>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}
