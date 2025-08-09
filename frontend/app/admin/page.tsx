'use client'

import AdminDashboardLayout from '@/components/admin-dashboard-layout';
import { Card, CardContent } from "@/components/ui/card";
import { BookOpen } from "lucide-react";

export default function AdminDashboardPage() {
  return (
    <AdminDashboardLayout>
      <div className="p-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-6">
          Admin Dashboard
        </h1>

        <Card>
          <CardContent className="p-8 text-center">
            <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No Updates Available
            </h3>
            <p className="text-gray-500">
              Updates will appear here when they are added.
            </p>
          </CardContent>
        </Card>
      </div>
    </AdminDashboardLayout>
  )
}

