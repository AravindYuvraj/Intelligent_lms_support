'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import DashboardLayout from '@/components/dashboard-layout'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Bookmark, RotateCcw } from 'lucide-react'

interface Message {
  id: string
  sender: string
  role: 'student' | 'admin' | 'agent'
  message: string
  timestamp: string
  avatar?: string
}

interface TicketDetail {
  id: string
  title: string
  status: 'Open' | 'Work in Progress' | 'Action Required' | 'Resolved' | 'Closed'
  category: string
  activity: string
  messages: Message[]
  rating?: number
}

export default function TicketDetailPage() {
  const params = useParams()
  const router = useRouter()
  const [ticket, setTicket] = useState<TicketDetail | null>(null)
  const [selectedRating, setSelectedRating] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Mock data - replace with actual API call
    const mockTicket: TicketDetail = {
      id: params.id as string,
      title: 'Update in Resume',
      status: 'Resolved',
      category: 'Mac',
      activity: 'Resume',
      messages: [
        {
          id: '1',
          sender: 'Kinjal Monaya',
          role: 'student',
          message: "Dear Team,\n\nI've changed my LinkedIn Link and updated a project in my resume. Request you to kindly update it in your record as well.\n\nFollowing is the drive link- https://drive.google.com/file/d/1GTJfwpwazibTeeGnx8D1OoeFZNGBAvew?usp=sharing\n\nThanks and Regards.",
          timestamp: '7 Sep, 2023 at 6:08 PM (IST)'
        },
        {
          id: '2',
          sender: 'Anamika Basu',
          role: 'admin',
          message: 'We were happy to assist you!\nYour ticket is now marked as resolved. We have sent you the feedback where you can share your the support experience.\nThank you\nStudent Experience Team',
          timestamp: '7 Sep, 2023 at 6:08 PM (IST)',
          avatar: 'AB'
        },
        {
          id: '3',
          sender: 'Anamika Basu',
          role: 'admin',
          message: 'Your resume has been updated',
          timestamp: '8 Sep, 2023 at 5:03 PM (IST)',
          avatar: 'AB'
        }
      ],
      rating: 4
    }
    
    setTicket(mockTicket)
    setSelectedRating(mockTicket.rating || null)
    setIsLoading(false)
  }, [params.id])

  const handleRating = async (rating: number) => {
    setSelectedRating(rating)
    
    // Mock API call to update rating
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/v1/tickets/${params.id}/rate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ rating })
      })
    } catch (error) {
      console.error('Failed to submit rating:', error)
    }
  }

  const handleReopenTicket = async () => {
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/v1/tickets/${params.id}/reopen`, {
        method: 'POST',
        credentials: 'include',
      })
      
      // Refresh the page or update state
      router.refresh()
    } catch (error) {
      console.error('Failed to reopen ticket:', error)
    }
  }

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

  const ratingEmojis = ['😠', '😞', '😐', '😊', '😍']

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="p-6 flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </DashboardLayout>
    )
  }

  if (!ticket) {
    return (
      <DashboardLayout>
        <div className="p-6">
          <div className="text-center">
            <h2 className="text-xl font-semibold text-gray-900">Ticket not found</h2>
            <p className="text-gray-600 mt-2">The ticket you're looking for doesn't exist.</p>
          </div>
        </div>
      </DashboardLayout>
    )
  }

  return (
    <DashboardLayout>
      <div className="p-6 max-w-4xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-semibold text-gray-900">
              {ticket.title}
            </h1>
            <Badge className={getStatusColor(ticket.status)}>
              {ticket.status.toUpperCase()}
            </Badge>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm">
              <Bookmark className="h-4 w-4 mr-2" />
              BOOKMARK
            </Button>
            
            {ticket.status === 'Resolved' && (
              <Button 
                variant="outline" 
                size="sm"
                onClick={handleReopenTicket}
                className="text-blue-600 border-blue-600 hover:bg-blue-50"
              >
                <RotateCcw className="h-4 w-4 mr-2" />
                REOPEN TICKET
              </Button>
            )}
          </div>
        </div>

        {/* Ticket Info */}
        <div className="mb-6 text-sm text-gray-600">
          <span>Support related to: </span>
          <span className="font-medium">{ticket.category}</span>
          <span className="mx-2">•</span>
          <span>Activity: </span>
          <span className="font-medium">{ticket.activity}</span>
        </div>

        {/* Messages */}
        <div className="space-y-6 mb-8">
          {ticket.messages.map((message, index) => (
            <Card key={message.id}>
              <CardHeader className="pb-3">
                <div className="flex items-center space-x-3">
                  <Avatar className="h-8 w-8">
                    <AvatarImage src={message.avatar ? `/placeholder.svg?height=32&width=32` : undefined} />
                    <AvatarFallback className="bg-green-100 text-green-700">
                      {message.avatar || message.sender.split(' ').map(n => n[0]).join('')}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <div className="font-medium text-gray-900">{message.sender}</div>
                    {index === 1 && (
                      <Badge className="bg-orange-100 text-orange-800 text-xs mt-1">
                        TICKET RESOLVED
                      </Badge>
                    )}
                  </div>
                  <div className="flex-1"></div>
                  <div className="text-sm text-gray-500">{message.timestamp}</div>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="text-gray-700 whitespace-pre-line">
                  {message.message}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Rating Section */}
        {ticket.status === 'Resolved' && (
          <Card>
            <CardContent className="p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Rating & Feedback</h3>
              <div className="mb-4">
                <p className="text-sm text-gray-600 mb-3">How happy are you with the assistance?</p>
                <div className="flex space-x-2">
                  {ratingEmojis.map((emoji, index) => {
                    const rating = index + 1
                    return (
                      <button
                        key={rating}
                        onClick={() => handleRating(rating)}
                        className={`text-2xl p-2 rounded-lg transition-all ${
                          selectedRating === rating
                            ? 'bg-blue-100 scale-110'
                            : 'hover:bg-gray-100 hover:scale-105'
                        }`}
                      >
                        {emoji}
                      </button>
                    )
                  })}
                </div>
              </div>
              
              {selectedRating && (
                <div className="text-sm text-gray-600">
                  Thank you for your feedback! You rated this resolution {selectedRating} out of 5.
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  )
}
