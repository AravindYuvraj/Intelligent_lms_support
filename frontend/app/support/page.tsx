'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import DashboardLayout from '@/components/dashboard-layout'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { MessageSquare, Star } from 'lucide-react'

interface Ticket {
  id: string
  title: string
  status: 'Open' | 'Work in Progress' | 'Action Required' | 'Resolved' | 'Closed'
  responseCount: number
  lastUpdated: string
  assignedTo: string
  rating?: number
}

export default function SupportPage() {
  const [activeTab, setActiveTab] = useState<'unresolved' | 'resolved'>('unresolved')
  const [tickets, setTickets] = useState<Ticket[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    // Mock data - replace with actual API call
    const mockTickets: Ticket[] = [
      {
        id: 'TKT-001',
        title: 'Update in Resume',
        status: 'Resolved',
        responseCount: 1,
        lastUpdated: '8 Sep, 2023 at 5:03 PM (IST)',
        assignedTo: 'Mac',
        rating: 4
      },
      {
        id: 'TKT-002',
        title: 'Portfolio Clearance',
        status: 'Closed',
        responseCount: 2,
        lastUpdated: '25 Aug, 2023 at 1:24 PM (IST)',
        assignedTo: 'Mac'
      },
      {
        id: 'TKT-003',
        title: 'Updating Final Resume',
        status: 'Resolved',
        responseCount: 1,
        lastUpdated: '17 Jul, 2023 at 9:23 AM (IST)',
        assignedTo: 'Mac',
        rating: 5
      },
      {
        id: 'TKT-004',
        title: 'RM Mock Assessment Result - 02',
        status: 'Resolved',
        responseCount: 1,
        lastUpdated: '8 Jul, 2023 at 1:24 PM (IST)',
        assignedTo: 'Revision'
      },
      {
        id: 'TKT-005',
        title: 'Marks Calculation',
        status: 'Resolved',
        responseCount: 1,
        lastUpdated: '3 May, 2023 at 8:14 PM (IST)',
        assignedTo: 'Evaluation-score'
      }
    ]
    
    setTickets(mockTickets)
    setIsLoading(false)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Resolved':
        return 'bg-green-100 text-green-800'
      case 'Closed':
        return 'bg-blue-100 text-blue-800'
      case 'Open':
        return 'bg-yellow-100 text-yellow-800'
      case 'Work in Progress':
        return 'bg-orange-100 text-orange-800'
      case 'Action Required':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const filteredTickets = tickets.filter(ticket => {
    if (activeTab === 'resolved') {
      return ticket.status === 'Resolved' || ticket.status === 'Closed'
    }
    return ticket.status !== 'Resolved' && ticket.status !== 'Closed'
  })

  const renderStars = (rating?: number) => {
    if (!rating) return <span className="text-gray-400">--</span>
    
    return (
      <div className="flex items-center space-x-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`h-4 w-4 ${
              star <= rating ? 'text-yellow-400 fill-current' : 'text-gray-300'
            }`}
          />
        ))}
      </div>
    )
  }

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="p-6 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </DashboardLayout>
    )
  }

  return (
    <DashboardLayout>
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-semibold text-gray-900">Support Tickets</h1>
          <Link href="/support/create">
            <Button className="bg-blue-600 hover:bg-blue-700">
              CREATE TICKET
            </Button>
          </Link>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 mb-6">
          <button
            onClick={() => setActiveTab('unresolved')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'unresolved'
                ? 'bg-gray-200 text-gray-900'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Unresolved
          </button>
          <button
            onClick={() => setActiveTab('resolved')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'resolved'
                ? 'bg-blue-100 text-blue-700'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Resolved
          </button>
        </div>

        {/* Tickets List */}
        <div className="space-y-4">
          {filteredTickets.length === 0 ? (
            <Card>
              <CardContent className="p-8 text-center">
                <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  No {activeTab} tickets
                </h3>
                <p className="text-gray-500">
                  {activeTab === 'resolved' 
                    ? "You don't have any resolved tickets yet."
                    : "You don't have any unresolved tickets."}
                </p>
              </CardContent>
            </Card>
          ) : (
            filteredTickets.map((ticket) => (
              <Card key={ticket.id} className="hover:shadow-md transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <Link 
                        href={`/support/ticket/${ticket.id}`}
                        className="text-lg font-medium text-gray-900 hover:text-blue-600 transition-colors"
                      >
                        {ticket.title}
                      </Link>
                      
                      <div className="flex items-center space-x-4 mt-2 text-sm text-gray-500">
                        <div className="flex items-center space-x-1">
                          <MessageSquare className="h-4 w-4" />
                          <span>{ticket.responseCount} Response</span>
                        </div>
                        <span>•</span>
                        <span>{ticket.assignedTo}</span>
                        <span>•</span>
                        <span>Last Updated on {ticket.lastUpdated}</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      {ticket.rating && (
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-gray-500">You rated</span>
                          {renderStars(ticket.rating)}
                        </div>
                      )}
                      
                      <Badge className={getStatusColor(ticket.status)}>
                        {ticket.status.toUpperCase()}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </div>
    </DashboardLayout>
  )
}
